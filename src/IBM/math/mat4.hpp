#ifndef IBM_MATH_MATRIX_HPP
#define IBM_MATH_MATRIX_HPP

#include <cmath>
#include <array>
#include <iostream>
#include <iomanip> 
#include "IBM/geometry/point.hpp"

namespace math {

struct Mat4 {
    std::array<std::array<float, 4>, 4> m;

    Mat4() {
        m = {{
            {{1.0f, 0.0f, 0.0f, 0.0f}},
            {{0.0f, 1.0f, 0.0f, 0.0f}},
            {{0.0f, 0.0f, 1.0f, 0.0f}},
            {{0.0f, 0.0f, 0.0f, 1.0f}}
        }};
    }

    static Mat4 identity() {
        return Mat4();
    }

    static Mat4 translation(float tx, float ty, float tz) {
        Mat4 t_mat = identity();
        t_mat.m[0][3] = tx;
        t_mat.m[1][3] = ty;
        t_mat.m[2][3] = tz;
        return t_mat;
    }

    static Mat4 scaling(float sx, float sy, float sz) {
        Mat4 s_mat = identity();
        s_mat.m[0][0] = sx;
        s_mat.m[1][1] = sy;
        s_mat.m[2][2] = sz;
        return s_mat;
    }

    static Mat4 rotationX(float angle_rad) {
        Mat4 r_mat = identity();
        float c = cosf(angle_rad);
        float s = sinf(angle_rad);
        r_mat.m[1][1] = c;
        r_mat.m[1][2] = -s;
        r_mat.m[2][1] = s;
        r_mat.m[2][2] = c;
        return r_mat;
    }

    static Mat4 rotationY(float angle_rad) {
        Mat4 r_mat = identity();
        float c = cosf(angle_rad);
        float s = sinf(angle_rad);
        r_mat.m[0][0] = c;
        r_mat.m[0][2] = s;
        r_mat.m[2][0] = -s;
        r_mat.m[2][2] = c;
        return r_mat;
    }

    static Mat4 rotationZ(float angle_rad) {
        Mat4 r_mat = identity();
        float c = cosf(angle_rad);
        float s = sinf(angle_rad);
        r_mat.m[0][0] = c;
        r_mat.m[0][1] = -s;
        r_mat.m[1][0] = s;
        r_mat.m[1][1] = c;
        return r_mat;
    }

    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 4; i++) {     
            for (int j = 0; j < 4; j++) { 
                result.m[i][j] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    result.m[i][j] += this->m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }

    geom::Point3D transformPoint(const geom::Point3D& p) const {
        geom::Point3D result_p;
        float px = p.x();
        float py = p.y();
        float pz = p.z();
        float pw = 1.0f;

        float rx = m[0][0] * px + m[0][1] * py + m[0][2] * pz + m[0][3] * pw;
        float ry = m[1][0] * px + m[1][1] * py + m[1][2] * pz + m[1][3] * pw;
        float rz = m[2][0] * px + m[2][1] * py + m[2][2] * pz + m[2][3] * pw;
        float rw = m[3][0] * px + m[3][1] * py + m[3][2] * pz + m[3][3] * pw;

        result_p[0] = rx;
        result_p[1] = ry;
        result_p[2] = rz;

        if (std::fabs(rw - 1.0f) > 1e-6f && std::fabs(rw) > 1e-6f) {
            result_p[0] /= rw;
            result_p[1] /= rw;
            result_p[2] /= rw;
        }
        return result_p;
    }

    geom::Point3D transformVector(const geom::Point3D& v) const {
        geom::Point3D result_v;
        float vx = v.x();
        float vy = v.y();
        float vz = v.z();

        result_v[0] = m[0][0] * vx + m[0][1] * vy + m[0][2] * vz;
        result_v[1] = m[1][0] * vx + m[1][1] * vy + m[1][2] * vz;
        result_v[2] = m[2][0] * vx + m[2][1] * vy + m[2][2] * vz;

        return result_v;
    }

    void print() const {
        std::cout << std::fixed << std::setprecision(3);
        for (int i = 0; i < 4; ++i) {
            std::cout << "[ ";
            for (int j = 0; j < 4; ++j) {
                std::cout << std::setw(8) << m[i][j] << (j == 3 ? "" : ", ");
            }
            std::cout << " ]" << std::endl;
        }
        std::cout << std::resetiosflags(std::ios_base::fixed) << std::setprecision(6);
    }
};

} // namespace math

#endif // !IBM_MATH_MATRIX_HPP
