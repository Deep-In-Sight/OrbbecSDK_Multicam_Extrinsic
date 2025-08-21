#include "libobsensor/ObSensor.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <thread>
#include <chrono>
#include <dirent.h>
#include <random>
#include <queue>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <limits>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "cJSON.h"
#include <libobsensor/hpp/Utils.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace ob;

// ArUco 마커 파라미터
const int ARUCO_DICT_ID = 16;  // DICT_16H5
const float ARUCO_MARKER_SIZE = 30.0f; // mm
const int EXPECTED_MARKERS = 4;  // 4개의 마커 (ID: 0,1,2,3)

static inline string nowTimestamp() {
	auto now      = std::chrono::system_clock::now();
	time_t t      = std::chrono::system_clock::to_time_t(now);
	struct tm tmv = {};
	localtime_r(&t, &tmv);
	char buf[32];
	strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
	return string(buf);
}

static inline double pointToPlaneDistance(const Vector3d &p, const Vector4d &plane) {
	// plane: ax + by + cz + d = 0, normalized a^2+b^2+c^2=1
	return fabs(plane.head<3>().dot(p) + plane(3));
}

static inline bool fitPlaneFrom3Points(const Vector3d &p1, const Vector3d &p2, const Vector3d &p3, Vector4d &planeOut) {
	Vector3d v1 = p2 - p1;
	Vector3d v2 = p3 - p1;
	Vector3d n  = v1.cross(v2);
	double    nrm = n.norm();
	if(nrm < 1e-6) return false;
	n.normalize();
	double d = -n.dot(p1);
	planeOut.head<3>() = n;
	planeOut(3)        = d;
	return true;
}

static inline void orderPolygonTLTRBRBL(vector<Point2f> &poly) {
	// Ensure TL, TR, BR, BL order for convex quad
	if(poly.size() != 4) return;
	// compute centroid
	Point2f c(0, 0);
	for(const auto &p: poly) c += p;
	c *= 0.25f;
	// separate by angle
	vector<pair<double, Point2f>> ang;
	for(const auto &p: poly) {
		double a = atan2(p.y - c.y, p.x - c.x);
		ang.push_back({ a, p });
	}
	sort(ang.begin(), ang.end(), [](auto &a, auto &b) { return a.first < b.first; });
	// angles: -pi..pi, we want TL(-135), TR(-45), BR(45), BL(135) if y-down image
	// map by y then x heuristic
	vector<Point2f> s;
	for(auto &ap: ang) s.push_back(ap.second);
	// reorder to TL, TR, BR, BL
	// find top-most two (smallest y)
	sort(s.begin(), s.end(), [](const Point2f &a, const Point2f &b) { return a.y < b.y || (a.y == b.y && a.x < b.x); });
	Point2f TL = (s[0].x < s[1].x) ? s[0] : s[1];
	Point2f TR = (s[0].x < s[1].x) ? s[1] : s[0];
	// bottom two
	Point2f BL = (s[2].x < s[3].x) ? s[2] : s[3];
	Point2f BR = (s[2].x < s[3].x) ? s[3] : s[2];
	poly.clear();
	poly.push_back(TL);
	poly.push_back(TR);
	poly.push_back(BR);
	poly.push_back(BL);
}

static vector<Point2f> computeInnerCorners(const map<int, vector<Point2f>> &markerCornerMap,
                                           const vector<int>              &ids) {
	// IDs expected: 0:TL, 1:TR, 2:BL, 3:BR
	if(markerCornerMap.size() < 4) return {};
	// marker centers
	map<int, Point2f> markerCenters;
	for(const auto &kv: markerCornerMap) {
		Point2f c(0, 0);
		for(const auto &p: kv.second) c += p;
		c *= 0.25f;
		markerCenters[kv.first] = c;
	}
	if(markerCenters.size() < 4) return {};
	// global centroid
	Point2f C(0, 0);
	for(const auto &kv: markerCenters) C += kv.second;
	C *= 0.25f;

	auto pickInner = [&](int id) -> Point2f {
		const auto &corners = markerCornerMap.at(id);
		Point2f m           = markerCenters[id];
		Point2f vin         = C - m;
		double   vinNorm     = sqrt(vin.x * vin.x + vin.y * vin.y);
		if(vinNorm < 1e-6) {
			// fallback: nearest to centroid
			double bestd = 1e18;
			Point2f best = corners[0];
			for(const auto &c: corners) {
				double dx = C.x - c.x, dy = C.y - c.y;
				double d2 = dx * dx + dy * dy;
				if(d2 < bestd) {
					bestd = d2;
					best  = c;
				}
			}
			return best;
		}
		vin.x /= vinNorm; vin.y /= vinNorm;
		double bestDot = -1e18;
		double bestD   = 1e18;
		Point2f best  = corners[0];
		for(const auto &c: corners) {
			Point2f u = c - m;
			double un = sqrt(u.x * u.x + u.y * u.y);
			if(un < 1e-6) continue;
			u.x /= un; u.y /= un;
			double dprod = u.x * vin.x + u.y * vin.y; // alignment with inward direction
			double dcen  = (C.x - c.x) * (C.x - c.x) + (C.y - c.y) * (C.y - c.y);
			// prefer larger dot; tie-break by nearer to centroid
			if(dprod > bestDot + 1e-6 || (fabs(dprod - bestDot) <= 1e-6 && dcen < bestD)) {
				bestDot = dprod;
				bestD   = dcen;
				best    = c;
			}
		}
		return best;
	};

	// assemble in TL, TR, BR, BL order from IDs 0,1,3,2
	vector<Point2f> quad(4);
	if(markerCornerMap.count(0) && markerCornerMap.count(1) && markerCornerMap.count(2) && markerCornerMap.count(3)) {
		quad[0] = pickInner(0); // TL
		quad[1] = pickInner(1); // TR
		quad[2] = pickInner(3); // BR
		quad[3] = pickInner(2); // BL
		orderPolygonTLTRBRBL(quad);
		return quad;
	}
	return {};
}

static Mat polygonMask(Size size, const vector<Point2f> &poly) {
	Mat mask = Mat::zeros(size, CV_8U);
	if(poly.size() != 4) return mask;
	vector<Point> pts;
	for(const auto &p: poly) pts.emplace_back(cvRound(p.x), cvRound(p.y));
	const Point *pp = pts.data();
	int          np = (int)pts.size();
	fillPoly(mask, &pp, &np, 1, Scalar(255));
	return mask;
}

// Depth->Color D2C 변환 폴백 (calibration2dTo2d를 이용해 직접 생성)
static shared_ptr<ob::Frame> buildDepthD2CByCalibration(const OBCalibrationParam &param,
	shared_ptr<ob::Frame> depthFrame,
	uint32_t colorW,
	uint32_t colorH) {
	if(!depthFrame) return nullptr;
	auto df = depthFrame->as<ob::DepthFrame>();
	if(!df) return nullptr;
	uint32_t dw = df->width();
	uint32_t dh = df->height();
	auto out = ob::FrameHelper::createFrame(OB_FRAME_DEPTH, df->format(), colorW, colorH, 0);
	if(!out) return nullptr;
	memset(out->data(), 0, colorW * colorH * 2);
	auto outData = (uint16_t *)out->data();
	const uint16_t *src = (const uint16_t *)df->data();
	float valueScale = df->getValueScale();

	// 더 견고한 경로: Depth 2D -> Depth 3D(undist) -> Color 2D
	for(uint32_t y = 0; y < dh; ++y) {
		for(uint32_t x = 0; x < dw; ++x) {
			uint16_t dRaw = src[y * dw + x];
			if(dRaw == 0) continue;
			float dmm = dRaw * valueScale; // mm

			OBPoint2f sp2d{ (float)x, (float)y };
			OBPoint3f p3d{};
			if(!ob::CoordinateTransformHelper::calibration2dTo3dUndistortion(param, sp2d, dmm, OB_SENSOR_DEPTH, OB_SENSOR_DEPTH, &p3d)) {
				continue;
			}
			OBPoint2f tp2d{};
			if(!ob::CoordinateTransformHelper::calibration3dTo2d(param, p3d, OB_SENSOR_DEPTH, OB_SENSOR_COLOR, &tp2d)) {
				continue;
			}
			int u = (int)round(tp2d.x);
			int v = (int)round(tp2d.y);
			if(u < 0 || u >= (int)colorW || v < 0 || v >= (int)colorH) continue;
			uint16_t &dst = outData[v * colorW + u];
			if(dst == 0 || dRaw < dst) dst = dRaw; // z-buffer(가까운 깊이 유지)
		}
	}
	return out;
}

// D2C 변환 함수는 직접 CoordinateTransformHelper 호출로 대체됨

static bool ransacPlaneOnROI(const Mat &mask, shared_ptr<Frame> depthD2C, const OBCalibrationParam &param, Vector4d &bestPlane,
                             Mat &inlierMask, int maxIters = 1000, double inlierThreshMm = 10.0) {
	inlierMask = Mat::zeros(mask.size(), CV_8U);
	int    W   = mask.cols, H = mask.rows;
	auto  raw  = (uint16_t *)depthD2C->data();
	if(!raw) return false;
	vector<Point> candidates;
	candidates.reserve(W * H);
	for(int y = 0; y < H; ++y) {
		const uchar *m = mask.ptr<uchar>(y);
		for(int x = 0; x < W; ++x) {
			if(m[x]) {
				uint16_t d = raw[y * W + x];
				if(d > 0) candidates.emplace_back(x, y);
			}
		}
	}
	if(candidates.size() < 100) return false;
	std::mt19937 rng((uint32_t)time(nullptr));
	std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
	int   bestInliers = -1;
	bool  found       = false;
	for(int it = 0; it < maxIters; ++it) {
		// sample 3 distinct points
		Point p1 = candidates[dist(rng)], p2 = candidates[dist(rng)], p3 = candidates[dist(rng)];
		if(p1 == p2 || p1 == p3 || p2 == p3) { --it; continue; }
		uint16_t d1 = raw[p1.y * W + p1.x], d2 = raw[p2.y * W + p2.x], d3 = raw[p3.y * W + p3.x];
		if(d1 == 0 || d2 == 0 || d3 == 0) { --it; continue; }
		OBPoint3f P1{}, P2{}, P3{};
		OBPoint2f s1{ (float)p1.x, (float)p1.y }, s2{ (float)p2.x, (float)p2.y }, s3{ (float)p3.x, (float)p3.y };
		CoordinateTransformHelper::calibration2dTo3d(param, s1, (float)d1, OB_SENSOR_COLOR, OB_SENSOR_COLOR, &P1);
		CoordinateTransformHelper::calibration2dTo3d(param, s2, (float)d2, OB_SENSOR_COLOR, OB_SENSOR_COLOR, &P2);
		CoordinateTransformHelper::calibration2dTo3d(param, s3, (float)d3, OB_SENSOR_COLOR, OB_SENSOR_COLOR, &P3);
		Vector4d plane;
		if(!fitPlaneFrom3Points({ P1.x, P1.y, P1.z }, { P2.x, P2.y, P2.z }, { P3.x, P3.y, P3.z }, plane)) continue;
		int inliers = 0;
		for(int k = 0; k < 500; ++k) { // subsample 500 pixels
			size_t idx = dist(rng);
			Point  p  = candidates[idx];
			uint16_t d = raw[p.y * W + p.x];
			if(d == 0) continue;
			OBPoint3f P{};
			OBPoint2f s{ (float)p.x, (float)p.y };
			CoordinateTransformHelper::calibration2dTo3d(param, s, (float)d, OB_SENSOR_COLOR, OB_SENSOR_COLOR, &P);
			if(pointToPlaneDistance({ P.x, P.y, P.z }, plane) <= inlierThreshMm) inliers++;
		}
		if(inliers > bestInliers) {
			bestInliers = inliers;
			bestPlane   = plane;
			found       = true;
		}
	}
	if(!found) return false;
	// build full inlier mask
	for(int y = 0; y < H; ++y) {
		uchar *m = inlierMask.ptr<uchar>(y);
		for(int x = 0; x < W; ++x) {
			if(mask.at<uchar>(y, x) == 0) { m[x] = 0; continue; }
			uint16_t d = raw[y * W + x];
			if(d == 0) { m[x] = 0; continue; }
			OBPoint3f P{};
			OBPoint2f s{ (float)x, (float)y };
			CoordinateTransformHelper::calibration2dTo3d(param, s, (float)d, OB_SENSOR_COLOR, OB_SENSOR_COLOR, &P);
			m[x] = (pointToPlaneDistance({ P.x, P.y, P.z }, bestPlane) <= inlierThreshMm) ? 255 : 0;
		}
	}
	return true;
}

static Mat floodPlaneByDepthEdge(const Mat &mask, shared_ptr<Frame> depthD2C, double edgeThreshMm = 50.0) {
	int W = mask.cols, H = mask.rows;
	Mat visited = Mat::zeros(H, W, CV_8U);
	Mat out     = Mat::zeros(H, W, CV_8U);
	auto raw    = (uint16_t *)depthD2C->data();
	if(!raw) return out;
	// seed: mask center
	Moments mo = moments(mask, true);
	Point   seed(mo.m10 / max(1.0, mo.m00), mo.m01 / max(1.0, mo.m00));
	if(seed.x < 0 || seed.x >= W || seed.y < 0 || seed.y >= H) return out;
	if(mask.at<uchar>(seed) == 0 || raw[seed.y * W + seed.x] == 0) return out;
	uint16_t seedD = raw[seed.y * W + seed.x];
	std::queue<Point> q;
	q.push(seed);
	visited.at<uchar>(seed) = 1;
	out.at<uchar>(seed)      = 255;
	auto inside = [&](int x, int y) { return x >= 0 && x < W && y >= 0 && y < H; };
	const int dx[4] = { 1, -1, 0, 0 };
	const int dy[4] = { 0, 0, 1, -1 };
	while(!q.empty()) {
		Point p = q.front(); q.pop();
		uint16_t pd = raw[p.y * W + p.x];
		for(int k = 0; k < 4; ++k) {
			int nx = p.x + dx[k], ny = p.y + dy[k];
			if(!inside(nx, ny)) continue;
			if(visited.at<uchar>(ny, nx)) continue;
			if(mask.at<uchar>(ny, nx) == 0) continue;
			uint16_t nd = raw[ny * W + nx];
			if(nd == 0) continue;
			if(fabs((double)nd - (double)pd) <= edgeThreshMm) {
				visited.at<uchar>(ny, nx) = 1;
				out.at<uchar>(ny, nx)      = 255;
				q.push({ nx, ny });
			}
		}
	}
	return out;
}

static void maskToRegionPoints(const Mat &mask, shared_ptr<Frame> depthD2C, const Mat &colorBGR, const OBCalibrationParam &param,
                               vector<OBColorPoint> &outPoints) {
	int W = mask.cols, H = mask.rows;
	auto raw = (uint16_t *)depthD2C->data();
	for(int y = 0; y < H; ++y) {
		const uchar *m = mask.ptr<uchar>(y);
		for(int x = 0; x < W; ++x) {
			if(!m[x]) continue;
			uint16_t d = raw[y * W + x];
			if(d == 0) continue;
			OBPoint3f P{};
			OBPoint2f s{ (float)x, (float)y };
			CoordinateTransformHelper::calibration2dTo3d(param, s, (float)d, OB_SENSOR_COLOR, OB_SENSOR_COLOR, &P);
			OBColorPoint cp{};
			cp.x = P.x; cp.y = P.y; cp.z = P.z;
			Vec3b bgr = colorBGR.at<Vec3b>(y, x);
			cp.b = bgr[0]; cp.g = bgr[1]; cp.r = bgr[2];
			outPoints.push_back(cp);
		}
	}
}

static void createFullRGBDPointCloud(const OBCalibrationParam &param, uint32_t colorWidth, uint32_t colorHeight,
                                     shared_ptr<Frame> depthD2C, shared_ptr<Frame> colorFrame,
                                     vector<OBColorPoint> &outPoints) {
	uint32_t tableSize = colorWidth * colorHeight * 2;
	unique_ptr<float[]> tables(new float[tableSize]);
	OBXYTables          xyTables{};
	if(!CoordinateTransformHelper::transformationInitXYTables(param, OB_SENSOR_COLOR, tables.get(), &tableSize, &xyTables)) {
		return;
	}
	uint32_t         pointcloudSize = colorWidth * colorHeight * sizeof(OBColorPoint);
	unique_ptr<uint8_t[]> buffer(new uint8_t[pointcloudSize]);
	OBColorPoint *   pointPixel = (OBColorPoint *)buffer.get();
	CoordinateTransformHelper::transformationDepthToRGBDPointCloud(&xyTables, depthD2C->data(), colorFrame->data(), pointPixel);
	int pointsSize = pointcloudSize / sizeof(OBColorPoint);
	outPoints.assign(pointPixel, pointPixel + pointsSize);
}

struct PointCloudData {
    vector<OBColorPoint>      points;
    vector<OBColorPoint>      regionPoints;  // ArUco 마커 영역 내의 포인트
    string                    deviceSerial;
    Mat                       colorImage;
    vector<Point2f>           arucoCorners;  // ArUco 마커의 모든 코너
    map<int, vector<Point2f>> arucoMarkerCorners;  // ArUco 마커별 코너
    vector<Point2f>           regionDefiningCorners;  // 4개 마커로 정의된 영역의 코너
    vector<int>               arucoIds;          // ArUco 마커 ID들
    bool                      arucoFound;               // ArUco 마커 검출 여부
    vector<Point2f>           planeCorners;  // 평면의 4개 코너 (마커 중심점들)
};

struct TransformationResult {
    Matrix4d transformMatrix;
    string sourceDevice;
    string targetDevice;
    double rmse;
};

// 색상 포인트 클라우드를 PLY 파일로 저장
void saveRGBPointsToPly(const vector<OBColorPoint> &points, const string &fileName) {
    FILE *fp = fopen(fileName.c_str(), "wb+");
    if(!fp) {
        throw std::runtime_error("Failed to open file for writing");
    }

    // 유효한 포인트만 카운트
    int validPointsCount = 0;
    static const auto min_distance = 1e-6;
    for(const auto& point : points) {
        if(fabs(point.x) >= min_distance || fabs(point.y) >= min_distance || fabs(point.z) >= min_distance) {
            validPointsCount++;
        }
    }

    // PLY 헤더 작성
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", validPointsCount);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");

    // 유효한 포인트 작성
    for(const auto& point : points) {
        if(fabs(point.x) >= min_distance || fabs(point.y) >= min_distance || fabs(point.z) >= min_distance) {
            fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", 
                    point.x, point.y, point.z, 
                    (int)point.r, (int)point.g, (int)point.b);
        }
    }

    fflush(fp);
    fclose(fp);
}

// ArUco 마커 검출 함수
bool detectArUcoMarkers(const Mat &image, vector<Point2f> &corners, vector<int> &ids, vector<Point2f> &planeCorners,
                        map<int, vector<Point2f>> &markerCornerMap, vector<Point2f> &regionCorners, bool visualize = false) {
    Mat gray;
    if(image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = image;
    }

    // ArUco 딕셔너리 생성
    Ptr<aruco::Dictionary>       dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(ARUCO_DICT_ID));
    Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();

    // 마커 검출
    vector<vector<Point2f>> detectedCorners;
    aruco::detectMarkers(gray, dictionary, detectedCorners, ids, parameters);

    std::cout << "검출된 ArUco 마커 수: " << ids.size() << std::endl;

    if(ids.size() >= EXPECTED_MARKERS) {
        vector<pair<int, vector<Point2f>>> markersWithIds;
        for(size_t i = 0; i < ids.size(); i++) {
            if(ids[i] >= 0 && ids[i] < EXPECTED_MARKERS) {
                markersWithIds.push_back({ ids[i], detectedCorners[i] });
            }
        }

        if(markersWithIds.size() >= EXPECTED_MARKERS) {
            sort(markersWithIds.begin(), markersWithIds.end(), [](const pair<int, vector<Point2f>> &a, const pair<int, vector<Point2f>> &b) {
                return a.first < b.first;
            });

            corners.clear();
            planeCorners.clear();
            markerCornerMap.clear();

            // 1. 마커 코너 및 중심점 계산
            map<int, Point2f> markerCenters;
            for(const auto &marker: markersWithIds) {
                corners.insert(corners.end(), marker.second.begin(), marker.second.end());
                markerCornerMap[marker.first] = marker.second;

                Point2f center(0, 0);
                for(const auto &corner: marker.second) center += corner;
                center *= 0.25f;
                markerCenters[marker.first] = center;
            }

            // 2. 평면 중심 계산
            Point2f planeCenter(0, 0);
            if(!markerCenters.empty()) {
                for(const auto &pair: markerCenters) {
                    planeCenter += pair.second;
                }
                planeCenter *= (1.0f / markerCenters.size());
            }

            // 3. 각 마커에 대해 평면 중심에 가장 가까운 코너를 찾아 planeCorners에 추가
            for(const auto &marker: markersWithIds) {
                const auto &currentMarkerCorners = marker.second;
                Point2f closestCorner            = currentMarkerCorners[0];
                double  minDistanceSq            = -1.0;

                for(const auto &corner: currentMarkerCorners) {
                    double dx         = corner.x - planeCenter.x;
                    double dy         = corner.y - planeCenter.y;
                    double distanceSq = dx * dx + dy * dy;
                    if(minDistanceSq < 0 || distanceSq < minDistanceSq) {
                        minDistanceSq = distanceSq;
                        closestCorner = corner;
                    }
                }
                planeCorners.push_back(closestCorner);
            }

            // 내부 코너 기반 폴리곤 산출 (ID 매핑: 0:TL,1:TR,2:BL,3:BR)
            regionCorners = computeInnerCorners(markerCornerMap, ids);

            if(visualize) {
                Mat vis = image.clone();
                aruco::drawDetectedMarkers(vis, detectedCorners, ids);

                for(size_t i = 0; i < planeCorners.size(); i++) {
                    circle(vis, planeCorners[i], 10, Scalar(0, 255, 0), 2);
                    putText(vis, to_string(i), planeCorners[i] + Point2f(15, 15), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                }

                if(regionCorners.size() == 4) {
                    for(size_t i = 0; i < regionCorners.size(); ++i) {
                        line(vis, regionCorners[i], regionCorners[(i + 1) % 4], Scalar(255, 0, 0), 2);
                    }
                }

                imshow("ArUco Marker Detection", vis);
                waitKey(500);
            }

            std::cout << "ArUco 마커 검출 성공! 마커 수: " << markersWithIds.size() << std::endl;
            return true;
        }
    }

    std::cout << "ArUco 마커 검출 실패! 예상 마커 수: " << EXPECTED_MARKERS << ", 실제: " << ids.size() << std::endl;
    return false;
}

// getArUcoCenterPoints() 제거: 평면 기반 캘리브레이션으로 전환

// SVD를 사용하여 변환 행렬 계산
Matrix4d computeTransformationSVD(const vector<Vector3d>& sourcePoints,
                                  const vector<Vector3d>& targetPoints) {
    if(sourcePoints.size() != targetPoints.size() || sourcePoints.size() < 3) {
        throw runtime_error("Invalid point sets for transformation computation");
    }

    // 중심점 계산
    Vector3d sourceCentroid = Vector3d::Zero();
    Vector3d targetCentroid = Vector3d::Zero();
    
    for(size_t i = 0; i < sourcePoints.size(); i++) {
        sourceCentroid += sourcePoints[i];
        targetCentroid += targetPoints[i];
    }
    sourceCentroid /= sourcePoints.size();
    targetCentroid /= targetPoints.size();

    // 중심 이동된 점들
    MatrixXd sourceC(3, sourcePoints.size());
    MatrixXd targetC(3, sourcePoints.size());
    
    for(size_t i = 0; i < sourcePoints.size(); i++) {
        sourceC.col(i) = sourcePoints[i] - sourceCentroid;
        targetC.col(i) = targetPoints[i] - targetCentroid;
    }

    // SVD를 사용한 회전 행렬 계산
    Matrix3d H = sourceC * targetC.transpose();
    JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixV() * svd.matrixU().transpose();

    // 반사 보정
    if(R.determinant() < 0) {
        Matrix3d V = svd.matrixV();
        V.col(2) *= -1;
        R = V * svd.matrixU().transpose();
    }

    // 이동 벡터 계산
    Vector3d t = targetCentroid - R * sourceCentroid;

    // 4x4 변환 행렬 구성
    Matrix4d T = Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;

    return T;
}

// 평면 PCD로부터 PCA 기반 초기 정합 행렬 계산 (Target -> Reference)
static bool computePlaneBasis(const vector<Vector3d>& pts, Vector3d& centroidOut, Matrix3d& basisOut) {
    if(pts.size() < 10) return false;
    centroidOut = Vector3d::Zero();
    for(const auto& p : pts) centroidOut += p;
    centroidOut /= (double)pts.size();
    Matrix3d cov = Matrix3d::Zero();
    for(const auto& p : pts) {
        Vector3d d = p - centroidOut;
        cov.noalias() += d * d.transpose();
    }
    SelfAdjointEigenSolver<Matrix3d> es(cov);
    if(es.info() != Success) return false;
    // eigenvalues ascending; evecs columns
    Vector3d n = es.eigenvectors().col(0);      // smallest eigenvalue -> plane normal
    Vector3d u = es.eigenvectors().col(2);      // largest eigenvalue  -> principal axis 1
    // ensure orthonormal right-handed basis
    n.normalize(); u.normalize();
    Vector3d v = n.cross(u); v.normalize();
    u = v.cross(n); u.normalize();
    basisOut.col(0) = u; basisOut.col(1) = v; basisOut.col(2) = n;
    return true;
}

static bool computeInitFromPlanePointClouds(const vector<Vector3d>& srcPts, const vector<Vector3d>& dstPts, Matrix4d& Tout) {
    Vector3d cs, cd; Matrix3d Bs, Bd;
    if(!computePlaneBasis(srcPts, cs, Bs)) return false;
    if(!computePlaneBasis(dstPts, cd, Bd)) return false;
    // align normal orientation
    if(Bs.col(2).dot(Bd.col(2)) < 0) {
        Bs.col(2) *= -1.0; Bs.col(1) *= -1.0; // keep right-handed
    }
    // rotation that maps src basis to dst basis
    Matrix3d R = Bd * Bs.transpose();
    if(R.determinant() < 0) {
        // fix possible reflection due to eigenvector sign ambiguity
        Matrix3d F = Matrix3d::Identity(); F(2,2) = -1.0; // flip normal axis
        R = Bd * F * Bs.transpose();
    }
    Vector3d t = cd - R * cs;
    Tout = Matrix4d::Identity();
    Tout.block<3,3>(0,0) = R;
    Tout.block<3,1>(0,3) = t;
    return true;
}

static double computeNearestNeighborRMSE(const vector<Vector3d>& srcPts, const vector<Vector3d>& dstPts, const Matrix4d& T, double maxDist = 50.0) {
    if(srcPts.empty() || dstPts.empty()) return -1.0;
    double maxDist2 = maxDist * maxDist;
    double sum = 0.0; int cnt = 0;
    for(const auto& p : srcPts) {
        Vector4d q(p.x(), p.y(), p.z(), 1.0);
        q = T * q;
        Vector3d ps = q.head<3>();
        double best = 1e18;
        for(const auto& d : dstPts) {
            double dd = (ps - d).squaredNorm();
            if(dd < best) best = dd;
        }
        if(best < maxDist2) { sum += best; cnt++; }
    }
    if(cnt == 0) return -1.0;
    return sqrt(sum / (double)cnt);
}

// 안정적인 프레임셋 획득 함수
shared_ptr<ob::FrameSet> getStableFrameset(shared_ptr<ob::Pipeline>& pipeline, 
                                          int maxRetries = 10, 
                                          int timeoutMs = 2000) {  // 타임아웃을 2초로 증가
    shared_ptr<ob::FrameSet> frameset = nullptr;
    int retries = 0;
    bool hasDepth = false;
    bool hasColor = false;
    
    cout << "프레임셋 획득 시작 (최대 " << maxRetries << "회 시도, 타임아웃: " << timeoutMs << "ms)" << endl;
    
    while(retries < maxRetries) {
        try {
            for(int i = 0; i < 20; i++) {
                frameset = pipeline->waitForFrames(timeoutMs); // auto exposure
            }
            
            if(frameset) {
                // 각 프레임 상태 체크
                hasDepth = (frameset->depthFrame() != nullptr);
                hasColor = (frameset->colorFrame() != nullptr);
                
                cout << "[시도 " << (retries + 1) << "] ";
                cout << "Depth: " << (hasDepth ? "OK" : "NO") << ", ";
                cout << "Color: " << (hasColor ? "OK" : "NO");
                
                if(hasDepth && hasColor) {
                    // 프레임 데이터 유효성 검사
                    auto depthFrame = frameset->depthFrame();
                    auto colorFrame = frameset->colorFrame();
                    
                    uint32_t depthSize = depthFrame->dataSize();
                    uint32_t colorSize = colorFrame->dataSize();
                    
                    cout << " | Depth size: " << depthSize << ", Color size: " << colorSize << endl;
                    
                    if(depthSize > 0 && colorSize > 0) {
                        cout << "프레임셋 획득 성공!" << endl;
                        return frameset;
                    } else {
                        cout << " - 데이터 크기가 0입니다." << endl;
                    }
                } else {
                    cout << " - 일부 프레임이 누락되었습니다." << endl;
                }
            } else {
                cout << "[시도 " << (retries + 1) << "] 프레임셋이 null입니다." << endl;
            }
        }
        catch(ob::Error &e) {
            cout << "[오류] " << e.getMessage() << endl;
        }
        catch(exception &e) {
            cout << "[예외] " << e.what() << endl;
        }
        
        retries++;
        if(retries < maxRetries) {
            cout << "재시도 전 대기 중... (500ms)" << endl;
            this_thread::sleep_for(chrono::milliseconds(500));
        }
    }
    
    cout << "프레임셋 획득 실패 (최대 재시도 횟수 초과)" << endl;
    return nullptr;
}

// 단일 디바이스 처리 함수
PointCloudData processSingleDevice(shared_ptr<ob::Device> device, 
                                  const vector<int>& meanFrameNums,
                                  bool saveFiles = true) {
    PointCloudData result;
    
    auto deviceInfo      = device->getDeviceInfo();
    result.deviceSerial  = string(deviceInfo->serialNumber());
    string ts            = nowTimestamp();
    
    cout << "\n=== 디바이스 처리 중: " << result.deviceSerial << " ===" << endl;
    
    // 파이프라인 생성
    auto pipeline = make_shared<ob::Pipeline>(device);
    
    // 스트림 설정
    auto config = make_shared<ob::Config>();

    std::shared_ptr<ob::VideoStreamProfile> colorProfile = nullptr;
    
    try {
        // Get all stream profiles of the color camera, including stream resolution, frame rate, and frame format
        auto colorProfiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);
        colorProfile       = colorProfiles->getVideoStreamProfile(3840, OB_HEIGHT_ANY, OB_FORMAT_RGB, OB_FPS_ANY);
        if(colorProfile) {
            std::cout << "colorProfile: " << colorProfile->width() << "x" << colorProfile->height() << std::endl;
            config->enableStream(colorProfile);
        }
    }
    catch(ob::Error &e) {
        config->setAlignMode(ALIGN_DISABLE);
        std::cerr << "Current device is not support color sensor!" << std::endl;
    }

    // Get all stream profiles of the depth camera, including stream resolution, frame rate, and frame format
    std::shared_ptr<ob::StreamProfileList> depthProfileList;
    OBAlignMode                            alignMode = ALIGN_DISABLE;
    
    depthProfileList = pipeline->getStreamProfileList(OB_SENSOR_DEPTH);


    if(depthProfileList->count() > 0) {
        std::shared_ptr<ob::StreamProfile> depthProfile;
        try {
            // Select the profile with the same frame rate as color.
            if(colorProfile) {
                depthProfile = depthProfileList->getVideoStreamProfile(512, OB_HEIGHT_ANY, OB_FORMAT_ANY, colorProfile->fps());
            }
        }
        catch(...) {
            depthProfile = nullptr;
        }

        if(!depthProfile) {
            // If no matching profile is found, select the default profile.
            depthProfile = depthProfileList->getProfile(OB_PROFILE_DEFAULT);
        }
        config->enableStream(depthProfile);
    }
    
    config->setAlignMode(alignMode);
    
    // 파이프라인 시작
    pipeline->start(config);
    
    // 워밍업
    cout << "디바이스 워밍업 중..." << endl;
    this_thread::sleep_for(chrono::milliseconds(1000));
    
    // 초기 프레임 획득 (안정적인 함수 사용)
    cout << "\n초기 프레임 획득 중..." << endl;
    shared_ptr<ob::FrameSet> init_frameset = getStableFrameset(pipeline, 20, 3000);  // 재시도 횟수와 타임아웃 증가
    
    if(!init_frameset || !init_frameset->depthFrame() || !init_frameset->colorFrame()) {
        cerr << "초기 프레임 획득 실패! 디바이스를 확인하세요." << endl;
        pipeline->stop();
        return result;
    }
    
    cout << "초기 프레임 획득 성공!" << endl;
    
    	// 포인트 클라우드 필터 생성
	auto pointCloudFilter = make_shared<ob::PointCloudFilter>();
	auto cameraParam      = pipeline->getCameraParam();
	pointCloudFilter->setCameraParam(cameraParam);
	auto calibParam       = pipeline->getCalibrationParam(config);

    
    
    auto init_depth_frame = init_frameset->depthFrame();
    auto init_color_frame = init_frameset->colorFrame();
    
    const uint32_t depth_width  = init_depth_frame->width();
    const uint32_t depth_height = init_depth_frame->height();
    const uint32_t depth_size   = depth_width * depth_height;

    cout << "Depth frame info: " << depth_width << "x" << depth_height << endl;

    const uint32_t color_width  = colorProfile ? colorProfile->width() : init_color_frame->width();
    const uint32_t color_height = colorProfile ? colorProfile->height() : init_color_frame->height();
    
    // 프레임 정보 디버그 출력
    cout << "Color frame info: " << color_width << "x" << color_height << endl;
    cout << "Color frame format: " << init_color_frame->format() << endl;
    cout << "Color frame data size: " << init_color_frame->dataSize() << endl;
    
    // 컬러 이미지를 OpenCV Mat으로 변환
    // 포맷에 따라 적절한 변환 수행
    if(init_color_frame->format() == OB_FORMAT_RGB) {
        result.colorImage = Mat(color_height, color_width, CV_8UC3, init_color_frame->data()).clone();
        cvtColor(result.colorImage, result.colorImage, COLOR_RGB2BGR);
    } else if(init_color_frame->format() == OB_FORMAT_BGR) {
        result.colorImage = Mat(color_height, color_width, CV_8UC3, init_color_frame->data()).clone();
    } else if(init_color_frame->format() == OB_FORMAT_YUYV) {
        Mat yuyv_image(color_height, color_width, CV_8UC2, init_color_frame->data());
        cvtColor(yuyv_image, result.colorImage, COLOR_YUV2BGR_YUYV);
    } else if(init_color_frame->format() == OB_FORMAT_I420) {
        Mat yuv_image(color_height * 3 / 2, color_width, CV_8UC1, init_color_frame->data());
        cvtColor(yuv_image, result.colorImage, COLOR_YUV2BGR_I420);
    } else if(init_color_frame->format() == OB_FORMAT_MJPG) {
        // MJPEG 디코딩
        vector<uint8_t> mjpeg_data((uint8_t*)init_color_frame->data(), 
                                   (uint8_t*)init_color_frame->data() + init_color_frame->dataSize());
        result.colorImage = imdecode(mjpeg_data, IMREAD_COLOR);
        if(result.colorImage.empty()) {
            cerr << "Failed to decode MJPEG frame!" << endl;
            pipeline->stop();
            return result;
        }
    } else {
        cerr << "Unsupported color format: " << init_color_frame->format() << endl;
        pipeline->stop();
        return result;
    }
    
    cout << "Mat created successfully with size: " << result.colorImage.cols << "x" << result.colorImage.rows << endl;

    // ArUco 마커 검출
    cout << "ArUco 마커 검출 중..." << endl;
    result.arucoFound = detectArUcoMarkers(result.colorImage, result.arucoCorners, result.arucoIds, result.planeCorners, result.arucoMarkerCorners,
										 result.regionDefiningCorners, true);

    if(!(result.arucoFound && result.regionDefiningCorners.size() == 4)) {
        cout << "ArUco 4개가 검출되지 않아 저장을 생략합니다." << endl;
        pipeline->stop();
        return result;
    }
    
    // 여러 프레임 수로 평균 계산
    map<int, vector<uint64_t>> depth_sums;
    map<int, vector<uint64_t>> depth_sum_sqs;
    map<int, vector<uint16_t>> depth_counts;
    
    for(int frameNum : meanFrameNums) {
        depth_sums[frameNum].resize(depth_size, 0);
        depth_sum_sqs[frameNum].resize(depth_size, 0);
        depth_counts[frameNum].resize(depth_size, 0);
    }
    
    // 최대 프레임 수만큼 수집
    int maxFrames = *max_element(meanFrameNums.begin(), meanFrameNums.end());
    cout << "프레임 수집 중 (최대 " << maxFrames << " 프레임)..." << endl;
    
    int collected = 0;
    while(collected < maxFrames) {
        auto frameset = pipeline->waitForFrames(100);
        if(frameset && frameset->depthFrame()) {
            auto depth_frame = frameset->depthFrame();
            if(depth_frame->width() == depth_width && depth_frame->height() == depth_height) {
                const uint16_t* depth_data = (const uint16_t*)depth_frame->data();
                
                for(auto frameNum : meanFrameNums) {
                    if(collected < frameNum) {
                        for(uint32_t i = 0; i < depth_size; i++) {
                            if(depth_data[i] > 0) {
                                depth_sums[frameNum][i] += depth_data[i];
                                depth_sum_sqs[frameNum][i] += (uint64_t)depth_data[i] * depth_data[i];
                                depth_counts[frameNum][i]++;
                            }
                        }
                    }
                }
                collected++;
                
                if(collected % 10 == 0) {
                    cout << "수집된 프레임: " << collected << "/" << maxFrames << endl;
                }
            }
        }
    }
    destroyAllWindows();
    
    // 평균 depth 계산용 프레임셋 획득
    shared_ptr<ob::FrameSet> target_frameset = getStableFrameset(pipeline, 5, 2000);
    if(!target_frameset) {
        cout << "템플릿 프레임 획득 실패, 종료..." << endl;
        pipeline->stop();
        return result;
    }

    // 평균 depth 계산
    uint16_t* out_depth_data = (uint16_t*)target_frameset->depthFrame()->data();
    for(uint32_t i = 0; i < depth_size; i++) {
        // 가장 긴 프레임 수 기준 평균 사용
        int frameNum = maxFrames;
        if(depth_counts[frameNum][i] > 0) {
            double mean     = (double)depth_sums[frameNum][i] / depth_counts[frameNum][i];
            double variance = (double)depth_sum_sqs[frameNum][i] / depth_counts[frameNum][i] - mean * mean;
            double std_dev  = sqrt(max(0.0, variance));
            if(std_dev < (mean * 0.2)) out_depth_data[i] = (uint16_t)round(mean);
            else out_depth_data[i] = 0;
        }
        else out_depth_data[i] = 0;
    }
	
    // SDK 포인트클라우드 생성 (depth only -> color 매핑)
    result.points.clear();
    pointCloudFilter->setCreatePointFormat(OB_FORMAT_POINT);
    auto pcFrame = pointCloudFilter->process(target_frameset->depthFrame());
    
    if(pcFrame) {
        OBPoint3f *points3D = (OBPoint3f *)pcFrame->data();

        int pointCnt = pcFrame->dataSize() / sizeof(OBPoint3f);
        for(int i = 0; i < pointCnt; ++i) {
            const auto &P = points3D[i]; if(P.z <= 0) continue;
            OBColorPoint cp{}; cp.x = P.x; cp.y = P.y; cp.z = P.z;
            OBPoint2f uv{};
            if(CoordinateTransformHelper::calibration3dTo2d(calibParam, { P.x, P.y, P.z }, OB_SENSOR_DEPTH, OB_SENSOR_COLOR, &uv)) {
                int u = (int)uv.x, v = (int)uv.y;
                if(u >= 0 && u < (int)color_width && v >= 0 && v < (int)color_height) {
                    Vec3b bgr = result.colorImage.at<Vec3b>(v, u);
                    cp.b = bgr[0]; cp.g = bgr[1]; cp.r = bgr[2];
                } else { cp.r = cp.g = cp.b = 255; }
            } else { cp.r = cp.g = cp.b = 255; }
            result.points.push_back(cp);
        }
    }

    // 1) RGB 내부 영역 마스크 생성 (regionDefiningCorners: TL,TR,BR,BL)
    Mat roiMaskColor = Mat::zeros(result.colorImage.size(), CV_8U);
    if(result.regionDefiningCorners.size() == 4) {
        vector<Point> poly;
        for(const auto &p : result.regionDefiningCorners) poly.emplace_back((int)round(p.x), (int)round(p.y));
        const Point *pp = poly.data(); int np = (int)poly.size();
        fillPoly(roiMaskColor, &pp, &np, 1, Scalar(255));
    }

    // 2) 전체 포인트클라우드 순회 → 3D를 Color로 투영 → ROI 내부만 plane 영역으로 수집
    result.regionPoints.clear();
    if(pcFrame && pcFrame->dataSize() > 0 && result.regionDefiningCorners.size() == 4) {
        int n = pcFrame->dataSize() / sizeof(OBPoint3f);
        const OBPoint3f *pts = (const OBPoint3f *)pcFrame->data();
        for(int i = 0; i < n; ++i) {
            const auto &P = pts[i]; if(P.z <= 0) continue;
            OBPoint2f uv{};
            if(!CoordinateTransformHelper::calibration3dTo2d(calibParam, { P.x, P.y, P.z }, OB_SENSOR_DEPTH, OB_SENSOR_COLOR, &uv)) continue;
            int u = (int)round(uv.x), v = (int)round(uv.y);
            if(u < 0 || u >= (int)color_width || v < 0 || v >= (int)color_height) continue;
            if(roiMaskColor.data && roiMaskColor.at<uchar>(v, u)) {
                OBColorPoint cp{}; cp.x = P.x; cp.y = P.y; cp.z = P.z;
                Vec3b bgr = result.colorImage.at<Vec3b>(v, u);
                cp.b = bgr[0]; cp.g = bgr[1]; cp.r = bgr[2];
                result.regionPoints.push_back(cp);
            }
        }
    }

    // 3) RANSAC 평면 피팅과 광선 교차를 통한 정밀 3D 코너 추정
    vector<Vector3d> centerPoints3D; // 최종 3D 코너 포인트
    if(result.regionPoints.size() >= 100) {
        // PCL 포인트 클라우드로 변환
        pcl::PointCloud<pcl::PointXYZ>::Ptr regionCloud(new pcl::PointCloud<pcl::PointXYZ>);
        regionCloud->points.reserve(result.regionPoints.size());
        for(const auto& p : result.regionPoints) {
            regionCloud->points.emplace_back(p.x, p.y, p.z);
        }

        // RANSAC으로 평면 모델 피팅
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(5.0); // 5mm
        seg.setInputCloud(regionCloud);
        seg.segment(*inliers, *coefficients);

        if (coefficients->values.size() >= 4) {
            cout << "RANSAC plane model fitted. Calculating corners by ray-plane intersection." << endl;
            Vector4d plane_coeffs(coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
            plane_coeffs /= plane_coeffs.head<3>().norm(); // 정규화

            // 2D 코너에서 3D 광선을 생성하고 평면과 교차시켜 정밀한 3D 코너 좌표 계산
            for (const auto& corner_2d : result.regionDefiningCorners) {
                OBPoint2f corner_ob = { corner_2d.x, corner_2d.y };
                OBPoint3f p1 = {}, p2 = {};

                // 2D 점을 깊이 값 두 개로 3D 공간(뎁스 센서 좌표계)에 투영하여 광선의 두 점을 생성
                bool p1_ok = CoordinateTransformHelper::calibration2dTo3d(calibParam, corner_ob, 1000.0f, OB_SENSOR_COLOR, OB_SENSOR_DEPTH, &p1);
                bool p2_ok = CoordinateTransformHelper::calibration2dTo3d(calibParam, corner_ob, 2000.0f, OB_SENSOR_COLOR, OB_SENSOR_DEPTH, &p2);

                if (p1_ok && p2_ok) {
                    Vector3d ray_origin(p1.x, p1.y, p1.z);
                    Vector3d ray_dir(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
                    ray_dir.normalize();

                    Vector3d plane_normal = plane_coeffs.head<3>();
                    double plane_d = plane_coeffs[3];

                    // 광선-평면 교차점 계산
                    double n_dot_d = plane_normal.dot(ray_dir);
                    if (std::abs(n_dot_d) > 1e-6) {
                        double t = (-plane_d - plane_normal.dot(ray_origin)) / n_dot_d;
                        Vector3d intersection_point = ray_origin + t * ray_dir;
                        centerPoints3D.push_back(intersection_point);
                    }
                }
            }
            if(centerPoints3D.size() != 4) {
                 cout << "Warning: Failed to calculate all 4 corner points via ray-plane intersection. Result might be inaccurate." << endl;
                 centerPoints3D.clear();
            } else {
                cout << "Successfully calculated 4 corner points on the robust plane." << endl;
            }
        } else {
            cout << "Warning: Could not estimate a plane model from the region points." << endl;
        }
    } else {
        cout << "Warning: Not enough points in plane region to perform robust corner estimation." << endl;
    }

    std::cout << "result.regionDefiningCorners: " << result.regionDefiningCorners.size() << std::endl;

	cout << "총 포인트 수: " << result.points.size() << endl;
	cout << "영역 내 포인트 수: " << result.regionPoints.size() << endl;

	// 파일 저장
	if(saveFiles && !result.points.empty()) {
        string prefix     = "capture_" + result.deviceSerial + "_" + ts;
        string fullPly    = prefix + "_full.ply";
        string regionPly  = "";
        saveRGBPointsToPly(result.points, fullPly);
        cout << "포인트 클라우드 저장됨: " << fullPly << endl;

        if(!result.regionPoints.empty()) {
            regionPly = prefix + "_plane.ply";
            saveRGBPointsToPly(result.regionPoints, regionPly);
            cout << "평면 영역 포인트 클라우드 저장됨: " << regionPly << endl;
        }

        // 컬러 이미지 저장
        string imgFilename = prefix + "_color.png";
        imwrite(imgFilename, result.colorImage);
        cout << "컬러 이미지 저장됨: " << imgFilename << endl;

        // ===== 디버그 시각화: 컬러 ROI 테두리만 저장 =====
        try {
            Mat visColor = result.colorImage.clone();
            vector<Point> colorPoly;
            for(const auto &p : result.regionDefiningCorners) colorPoly.emplace_back((int)round(p.x), (int)round(p.y));
            if(colorPoly.size() == 4) {
                polylines(visColor, colorPoly, true, Scalar(255, 0, 0), 2, LINE_AA);
            }
            string roiColorDbg = prefix + "_color_roi_debug.png";
            imwrite(roiColorDbg, visColor);
        } catch(...) { }

        // JSON 메타데이터 저장: SVD 초기값(마커 중심 3D) + 전체/평면 PLY 경로 저장
        {
            cJSON *root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "deviceSerial", result.deviceSerial.c_str());
            cJSON_AddStringToObject(root, "timestamp", ts.c_str());
            cJSON_AddBoolToObject(root, "arucoFound", result.arucoFound);
            cJSON_AddStringToObject(root, "fullPly", fullPly.c_str());
            if(!regionPly.empty()) cJSON_AddStringToObject(root, "regionPly", regionPly.c_str());
            cJSON_AddStringToObject(root, "colorImage", imgFilename.c_str());

            if(!centerPoints3D.empty()) {
                cJSON *pointsArray = cJSON_CreateArray();
                for(const auto &p: centerPoints3D) {
                    cJSON *point = cJSON_CreateObject();
                    cJSON_AddNumberToObject(point, "x", p.x());
                    cJSON_AddNumberToObject(point, "y", p.y());
                    cJSON_AddNumberToObject(point, "z", p.z());
                    cJSON_AddItemToArray(pointsArray, point);
                }
                cJSON_AddItemToObject(root, "arucoCenterPoints3D", pointsArray);
            }

            char *jsonString   = cJSON_Print(root);
            string jsonFilename = prefix + ".json";
            ofstream jsonFile(jsonFilename);
            jsonFile << jsonString;
            jsonFile.close();

            cout << "메타데이터 저장됨: " << jsonFilename << endl;

            free(jsonString);
            cJSON_Delete(root);
        }
        pipeline->stop();
    }
    
    return result;
}

// 메인 함수
int main(int argc, char **argv) try {
    ob::Context ctx;
    ctx.setLoggerSeverity(OB_LOG_SEVERITY_WARN);
    
    // 디바이스 목록 획득
    auto devList = ctx.queryDeviceList();
    uint32_t deviceCount = devList->deviceCount();
    
    cout << "\n=== Orbbec 체커보드 캘리브레이션 시스템 ===" << endl;
    if(deviceCount == 0) {
        cout << "연결된 디바이스가 없습니다! 파일 기반 캘리브레이션('c')만 가능합니다." << endl;
    }
    else {
        cout << "발견된 디바이스 수: " << deviceCount << endl << endl;
        // 디바이스 목록 표시
        for(uint32_t i = 0; i < deviceCount; i++) {
            auto dev = devList->getDevice(i);
            auto info = dev->getDeviceInfo();
            cout << "[" << i << "] " << info->name() << " - Serial: " << info->serialNumber() << endl;
        }
    }
    
    // 평균을 계산할 프레임 수들
    vector<int> meanFrameNums = {20, 30, 40};
    
    // 처리된 디바이스 데이터 저장
    vector<PointCloudData> processedDevices;
    
    // 사용자 입력 루프
    while(true) {
        cout << "\n명령어:" << endl;
        if(deviceCount > 0) {
            cout << "  [0-" << (deviceCount - 1) << "] : 해당 인덱스의 디바이스 처리" << endl;
            cout << "  a : 모든 디바이스 순차 처리" << endl;
        }
        cout << "  c : 수집된 데이터로 변환 행렬 계산" << endl;
        cout << "  q : 종료" << endl;
        cout << "선택: ";
        
        string input;
        cin >> input;
        
        if(input == "q") {
            break;
        }
        else if(input == "a") {
            if(deviceCount == 0) {
                cout << "연결된 디바이스가 없어 'a' 명령을 실행할 수 없습니다." << endl;
                continue;
            }
            // 모든 디바이스 처리
            processedDevices.clear();
            for(uint32_t i = 0; i < deviceCount; i++) {
                cout << "\n디바이스 " << i << " 처리 시작..." << endl;
                auto dev = devList->getDevice(i);
                PointCloudData data = processSingleDevice(dev, meanFrameNums);
                processedDevices.push_back(data);
                
                // 디바이스 간 딜레이
                this_thread::sleep_for(chrono::milliseconds(200));
            }
            cout << "\n모든 디바이스 처리 완료!" << endl;
        }
        else if(input == "c") {
            // 변환 행렬 계산 (파일 기반)
            struct CalibEntry { string serial; string timestamp; vector<Vector3d> innerCornerPoints3D; string regionPly; string fullPly; };
            vector<CalibEntry> entries;

            // 1. 현재 폴더에서 "capture_*.json" 파일 스캔
            DIR *          dir;
            struct dirent *ent;
            if((dir = opendir(".")) != NULL) {
                while((ent = readdir(dir)) != NULL) {
                    string filename = ent->d_name;
                    if(filename.rfind("capture_", 0) == 0 && filename.size() > 8 && filename.find(".json") != string::npos) {
                        ifstream jsonFile(filename);
                        if(!jsonFile.is_open()) continue;
                        string jsonString((istreambuf_iterator<char>(jsonFile)), istreambuf_iterator<char>());
                        jsonFile.close();

                        cJSON *root = cJSON_Parse(jsonString.c_str());
                        if(!root) continue;
                        cJSON *serial = cJSON_GetObjectItem(root, "deviceSerial");
                        cJSON *ts     = cJSON_GetObjectItem(root, "timestamp");
                        cJSON *found  = cJSON_GetObjectItem(root, "arucoFound");
                        cJSON *pts    = cJSON_GetObjectItem(root, "arucoCenterPoints3D");
                        cJSON *rply   = cJSON_GetObjectItem(root, "regionPly");
                        cJSON *fply   = cJSON_GetObjectItem(root, "fullPly");
                        if(cJSON_IsString(serial) && cJSON_IsString(ts) && cJSON_IsBool(found) && found->valueint && cJSON_IsArray(pts) && cJSON_GetArraySize(pts) >= 3) {
                            CalibEntry ce; ce.serial = serial->valuestring; ce.timestamp = ts->valuestring; if(cJSON_IsString(rply)) ce.regionPly = rply->valuestring; if(cJSON_IsString(fply)) ce.fullPly = fply->valuestring;
                            int n = cJSON_GetArraySize(pts);
                            for(int i = 0; i < n; ++i) {
                                cJSON *p = cJSON_GetArrayItem(pts, i);
                                cJSON *xx = cJSON_GetObjectItem(p, "x"); cJSON *yy = cJSON_GetObjectItem(p, "y"); cJSON *zz = cJSON_GetObjectItem(p, "z");
                                if(cJSON_IsNumber(xx) && cJSON_IsNumber(yy) && cJSON_IsNumber(zz)) ce.innerCornerPoints3D.push_back(Vector3d(xx->valuedouble, yy->valuedouble, zz->valuedouble));
                            }
                            entries.push_back(ce);
                        }
                        cJSON_Delete(root);
                    }
                }
                closedir(dir);
            }
            else { cerr << "현재 디렉토리를 열 수 없습니다!" << endl; continue; }

            if(entries.size() < 2) { cout << "유효한 캡처 파일이 2개 이상 필요합니다." << endl; continue; }

            // 정렬: serial, timestamp
            sort(entries.begin(), entries.end(), [](const CalibEntry &a, const CalibEntry &b) {
                if(a.serial == b.serial) return a.timestamp < b.timestamp;
                return a.serial < b.serial;
            });

            cout << "\n=== 변환 대상 선택 ===" << endl;
            for(size_t i = 0; i < entries.size(); ++i) {
                cout << "  [" << i << "] " << entries[i].serial << "  " << entries[i].timestamp;
                if(!entries[i].fullPly.empty()) cout << "  (full ply)";
                if(!entries[i].regionPly.empty()) cout << "  (plane ply)";
                cout << endl;
            }

            int refIndex = -1, targetIndex = -1;
            cout << "\n기준(Base) 인덱스: "; cin >> refIndex;
            cout << "대상(Target) 인덱스: "; cin >> targetIndex;
            if(refIndex < 0 || refIndex >= (int)entries.size() || targetIndex < 0 || targetIndex >= (int)entries.size() || refIndex == targetIndex) {
                cout << "잘못된 선택입니다." << endl; continue;
            }

            const auto &ref    = entries[refIndex];
            const auto &target = entries[targetIndex];

            auto loadPly = [](const string &path) {
                vector<Vector3d> pts; pts.reserve(50000);
                ifstream f(path); if(!f.is_open()) return pts;
                string line; bool header = true; int vertexCount = 0;
                while(getline(f, line)) {
                    if(header) {
                        if(line.rfind("element vertex", 0) == 0) {
                            sscanf(line.c_str(), "element vertex %d", &vertexCount);
                        }
                        if(line == "end_header") header = false;
                        continue;
                    }
                    if(vertexCount > 0 && (int)pts.size() >= vertexCount) break;
                    double x, y, z; std::istringstream iss(line);
                    if(!(iss >> x >> y >> z)) continue;
                    pts.emplace_back(x, y, z);
                    if((int)pts.size() >= 50000) break; // Safety limit
                }
                return pts;
            };

            // Load PLY files for ICP: prefer region (plane) points, fallback to full point cloud
            vector<Vector3d> pref, ptgt;
            bool useRegionPly = false;
            if(!ref.regionPly.empty() && !target.regionPly.empty()) {
                cout << "Attempting to load region PLY files for ICP..." << endl;
                pref = loadPly(ref.regionPly);
                ptgt = loadPly(target.regionPly);
                if(pref.size() >= 100 && ptgt.size() >= 100) {
                    cout << "Successfully loaded region PLY files." << endl;
                    useRegionPly = true;
                } else {
                    cout << "Region PLY files are too small or failed to load. Falling back to full PLY files." << endl;
                }
            }

            if(!useRegionPly) {
                cout << "Loading full PLY files for ICP..." << endl;
                if(ref.fullPly.empty() || target.fullPly.empty()) {
                    cout << "Selected items do not have full PLY paths." << endl;
                    continue;
                }
                pref = loadPly(ref.fullPly);
                ptgt = loadPly(target.fullPly);
            }

            if(pref.size() < 100 || ptgt.size() < 100) {
                cout << "Not enough points in the point clouds to proceed with ICP." << endl;
                continue;
            }

            // Initial Alignment: Prioritize SVD on corner points, fallback to PCA on plane point clouds.
            Matrix4d T = Matrix4d::Identity();
            bool initOk = false;

            if(ref.innerCornerPoints3D.size() >= 4 && ref.innerCornerPoints3D.size() == target.innerCornerPoints3D.size()) {
                cout << "Performing initial alignment using SVD on inner corner points..." << endl;
                try {
                    T = computeTransformationSVD(target.innerCornerPoints3D, ref.innerCornerPoints3D);
                    initOk = true;
                    cout << "SVD initial alignment successful." << endl;
                } catch(...) {
                    initOk = false;
                    cout << "SVD initial alignment failed. Falling back to PCA..." << endl;
                }
            }

            if(!initOk && useRegionPly) {
                cout << "Performing initial alignment using PCA on plane point clouds as fallback..." << endl;
                if(computeInitFromPlanePointClouds(ptgt, pref, T)) {
                    initOk = true;
                    cout << "PCA initial alignment successful." << endl;
                } else {
                    cout << "PCA initial alignment also failed." << endl;
                }
            }

            if(!initOk) {
                cout << "Initial alignment failed. Cannot proceed with ICP." << endl;
                continue;
            }

            // ICP 정련 (전체 포인트클라우드 사용)
            auto downsample = [](vector<Vector3d> &pts, size_t maxN) {
                if(pts.size() <= maxN) return;
                std::mt19937 rng((uint32_t)time(nullptr));
                std::shuffle(pts.begin(), pts.end(), rng);
                pts.resize(maxN);
            };

            auto refineICP = [&](const vector<Vector3d> &src0, const vector<Vector3d> &dst0, Matrix4d &Tinout,
                                 const vector<Vector3d>& target_corners, const vector<Vector3d>& ref_corners) {
                // 1. Data Preparation
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
                cloud_in->points.reserve(src0.size());
                for(const auto &p: src0) cloud_in->points.emplace_back(p.x(), p.y(), p.z());

                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
                cloud_target->points.reserve(dst0.size());
                for(const auto &p: dst0) cloud_target->points.emplace_back(p.x(), p.y(), p.z());

                cout << "Starting Iterative PCL Point-to-Point ICP refinement..." << endl;
                cout << "Source points: " << cloud_in->size() << ", Target points: " << cloud_target->size() << endl;

                Matrix4f current_transform = Tinout.cast<float>();
                Matrix4f best_transform = current_transform;

                auto calculate_rmse = [&](const Matrix4f& transform) {
                    if (target_corners.empty() || ref_corners.empty() || target_corners.size() != ref_corners.size()) return -1.0;
                    double rmse_sum = 0.0;
                    for (size_t j = 0; j < target_corners.size(); j++) {
                        Eigen::Vector4f p_target_h(target_corners[j](0), target_corners[j](1), target_corners[j](2), 1.0);
                        Eigen::Vector4f p_ref_h = transform * p_target_h;
                        Eigen::Vector3d diff = p_ref_h.head<3>().cast<double>() - ref_corners[j];
                        rmse_sum += diff.squaredNorm();
                    }
                    return sqrt(rmse_sum / target_corners.size());
                };

                double min_rmse = calculate_rmse(best_transform);
                if(min_rmse >= 0) {
                    cout << "Initial Corner RMSE: " << min_rmse << " mm" << endl;
                }

                // 2. Iterative Point-to-Point ICP
                pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                icp.setInputSource(cloud_in);
                icp.setInputTarget(cloud_target);
                icp.setMaximumIterations(300);
                icp.setTransformationEpsilon(1e-9);
                icp.setEuclideanFitnessEpsilon(1e-5);

                // Start with a small correspondence distance for fine-tuning an already good alignment
                vector<double> correspondence_distances = {10.0, 8.0, 6.0, 5.0, 4.0, 3.0};
                for(size_t i = 0; i < correspondence_distances.size(); ++i) {
                    cout << "  - ICP iteration " << i + 1 << " with correspondence distance: " << correspondence_distances[i] << "mm" << endl;
                    icp.setMaxCorrespondenceDistance(correspondence_distances[i]);
                    
                    pcl::PointCloud<pcl::PointXYZ> final_cloud;
                    icp.align(final_cloud, current_transform);
                    
                    if(icp.hasConverged()) {
                        current_transform = icp.getFinalTransformation();
                        double current_rmse = calculate_rmse(current_transform);
                        
                        cout << "    Converged. Score: " << icp.getFitnessScore() << ". Corner RMSE: " << current_rmse << " mm";
                        
                        if (current_rmse >= 0 && current_rmse < min_rmse) {
                            min_rmse = current_rmse;
                            best_transform = current_transform;
                            cout << " => New best RMSE found!" << endl;
                        } else {
                            cout << endl;
                        }
                    } else {
                        cout << "    Iteration FAILED to converge." << endl;
                        break; 
                    }
                }

                Tinout = best_transform.cast<double>();
                cout << "\nICP refinement finished. Best Corner RMSE: " << min_rmse << " mm" << endl;
            };

            // ICP 정련 (전체 포인트클라우드 사용)
            refineICP(ptgt, pref, T, target.innerCornerPoints3D, ref.innerCornerPoints3D);

            // 결과 출력 및 저장
            cout << "\n변환 행렬 (Target -> Reference):" << endl;
            cout << T << endl;

            // RMSE 계산 (중심점 기반)
            double rmse = 0; for(size_t j = 0; j < target.innerCornerPoints3D.size(); j++) { Vector4d p(target.innerCornerPoints3D[j](0), target.innerCornerPoints3D[j](1), target.innerCornerPoints3D[j](2), 1); Vector4d q = T * p; Vector3d diff = q.head<3>() - ref.innerCornerPoints3D[j]; rmse += diff.squaredNorm(); } rmse = sqrt(rmse / std::max<size_t>(1, target.innerCornerPoints3D.size()));
            cout << "RMSE: " << rmse << " mm" << endl;

            string filename = "Transform_" + target.serial + "_to_" + ref.serial + "_" + nowTimestamp() + ".txt";
            ofstream file(filename);
            if(file.is_open()) {
                file << "# Transformation Matrix from " << target.serial << " to " << ref.serial << "\n";
                file << "# RMSE: " << rmse << " mm\n";
                file << T << "\n";
                file.close();
                cout << "변환 행렬 저장됨: " << filename << endl;
            }
        }
        else {
            if(deviceCount > 0) {
                // 개별 디바이스 처리
                try {
                    int deviceIndex = stoi(input);
                    if(deviceIndex >= 0 && deviceIndex < (int)deviceCount) {
                        cout << "\n디바이스 " << deviceIndex << " 처리 시작..." << endl;
                        auto dev = devList->getDevice(deviceIndex);
                        auto data = processSingleDevice(dev, meanFrameNums);

                        // 이미 처리된 디바이스인지 확인
                        bool found = false;
                        for(auto &pd: processedDevices) {
                            if(pd.deviceSerial == data.deviceSerial) {
                                pd    = data; // 업데이트
                                found = true;
                                break;
                            }
                        }
                        if(!found) {
                            processedDevices.push_back(data);
                        }

                        cout << "디바이스 " << deviceIndex << " 처리 완료!" << endl;
                    }
                    else {
                        cout << "잘못된 디바이스 인덱스입니다!" << endl;
                    }
                }
                catch(...) {
                    cout << "잘못된 입력입니다!" << endl;
                }
            }
            else {
                cout << "잘못된 입력입니다! 'c' 또는 'q'를 입력하세요." << endl;
            }
        }
    }
    
    cout << "\n프로그램을 종료합니다." << endl;
    return 0;
}
catch(ob::Error &e) {
    cerr << "OrbbecSDK Error: " << e.getMessage() << endl;
    cerr << "Function: " << e.getName() << endl;
    cerr << "Args: " << e.getArgs() << endl;
    cerr << "Type: " << e.getExceptionType() << endl;
    return -1;
}
catch(exception &e) {
    cerr << "Standard Exception: " << e.what() << endl;
    return -1;
}