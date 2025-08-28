#pragma once

#include "quat.h"
#include "transform.h"

typedef struct i3_cam_t
{
    i3_vec3_t position;
    i3_vec3_t direction;
    i3_vec3_t up;
    float fov_y;
    float z_near;
    float z_far;
} i3_cam_t;

static inline void i3_cam(i3_cam_t* cam,
                          i3_vec3_t position,
                          i3_vec3_t direction,
                          i3_vec3_t up,
                          float fov_y_degree,
                          float z_near,
                          float z_far);
static inline void i3_cam_init_target(i3_cam_t* cam,
                                      i3_vec3_t position,
                                      i3_vec3_t target,
                                      i3_vec3_t up,
                                      float fov_y_degree,
                                      float z_near,
                                      float z_far);
static inline i3_vec3_t i3_cam_get_right(i3_cam_t* cam);
static inline void i3_cam_set_position(i3_cam_t* cam, i3_vec3_t position);
static inline i3_vec3_t i3_cam_get_position(i3_cam_t* cam);
static inline void i3_cam_set_direction(i3_cam_t* cam, i3_vec3_t direction);
static inline void i3_cam_set_target(i3_cam_t* cam, i3_vec3_t target);
static inline void i3_cam_yaw(i3_cam_t* cam, float angle);
static inline void i3_cam_pitch(i3_cam_t* cam, float angle);
static inline void i3_cam_roll(i3_cam_t* cam, float angle);
static inline void i3_cam_fly_forward(i3_cam_t* cam, float step);
static inline void i3_cam_fly_right(i3_cam_t* cam, float step);
static inline void i3_cam_fly_up(i3_cam_t* cam, float step);
static inline i3_mat4_t i3_cam_get_projection_matrix(i3_cam_t* cam, float aspect);
static inline i3_mat4_t i3_cam_get_view_matrix(i3_cam_t* cam);
static inline i3_mat4_t i3_cam_get_projection_view_matrix(i3_cam_t* cam, float aspect);

// cam implementation

static inline void i3_cam(i3_cam_t* cam,
                          i3_vec3_t position,
                          i3_vec3_t direction,
                          i3_vec3_t up,
                          float fov_y_degree,
                          float z_near,
                          float z_far)
{
    assert(cam != NULL);
    cam->position = position;
    cam->direction = i3_vec3_normalize(direction);
    cam->up = i3_vec3_normalize(up);
    cam->fov_y = i3_deg_to_radf(fov_y_degree);
    cam->z_near = z_near;
    cam->z_far = z_far;
}

static inline void i3_cam_init_target(i3_cam_t* cam,
                                      i3_vec3_t position,
                                      i3_vec3_t target,
                                      i3_vec3_t up,
                                      float fov_y_degree,
                                      float z_near,
                                      float z_far)
{
    assert(cam != NULL);

    // i3_vec3_t direction = i3_vec3_sub(target, position);
    // TODO: this is fishy ...
    i3_vec3_t direction = i3_vec3_sub(position, target);

    i3_cam(cam, position, direction, up, fov_y_degree, z_near, z_far);
}

static inline i3_vec3_t i3_cam_get_right(i3_cam_t* cam)
{
    return i3_vec3_cross(cam->direction, cam->up);
}

static inline void i3_cam_set_position(i3_cam_t* cam, i3_vec3_t position)
{
    assert(cam != NULL);
    cam->position = position;
}

static inline i3_vec3_t i3_cam_get_position(i3_cam_t* cam)
{
    assert(cam != NULL);
    return cam->position;
}

static inline void i3_cam_set_direction(i3_cam_t* cam, i3_vec3_t direction)
{
    assert(cam != NULL);
    cam->direction = i3_vec3_normalize(direction);
}

static inline void i3_cam_set_target(i3_cam_t* cam, i3_vec3_t target)
{
    assert(cam != NULL);
    cam->direction = i3_vec3_normalize(i3_vec3_sub(target, cam->position));
}

static inline void i3_cam_yaw(i3_cam_t* cam, float angle)
{
    assert(cam != NULL);

    i3_quat_t q = i3_quat_axis_angle(cam->up, angle);
    cam->direction = i3_quat_transform(q, cam->direction);
}

static inline void i3_cam_pitch(i3_cam_t* cam, float angle)
{
    assert(cam != NULL);

    i3_quat_t q = i3_quat_axis_angle(i3_cam_get_right(cam), angle);
    cam->up = i3_quat_transform(q, cam->up);
    cam->direction = i3_quat_transform(q, cam->direction);
}

static inline void i3_cam_roll(i3_cam_t* cam, float angle)
{
    assert(cam != NULL);

    i3_quat_t q = i3_quat_axis_angle(cam->direction, angle);
    cam->up = i3_quat_transform(q, cam->up);
}

static inline void i3_cam_fly_forward(i3_cam_t* cam, float step)
{
    assert(cam != NULL);

    cam->position = i3_vec3_add(cam->position, i3_vec3_scale(cam->direction, step));
}

static inline void i3_cam_fly_right(i3_cam_t* cam, float step)
{
    assert(cam != NULL);

    cam->position = i3_vec3_add(cam->position, i3_vec3_scale(i3_cam_get_right(cam), step));
}

static inline void i3_cam_fly_up(i3_cam_t* cam, float step)
{
    assert(cam != NULL);

    cam->position = i3_vec3_add(cam->position, i3_vec3_scale(cam->up, step));
}

static inline i3_mat4_t i3_cam_get_projection_matrix(i3_cam_t* cam, float aspect)
{
    assert(cam != NULL);
    return i3_mat4_persective_fov_rh(cam->fov_y, aspect, cam->z_near, cam->z_far);
}

static inline i3_mat4_t i3_cam_get_view_matrix(i3_cam_t* cam)
{
    assert(cam != NULL);
    return i3_mat4_look_to_rh(cam->position, cam->direction, cam->up);
}

static inline i3_mat4_t i3_cam_get_projection_view_matrix(i3_cam_t* cam, float aspect)
{
    return i3_mat4_mult(i3_cam_get_projection_matrix(cam, aspect), i3_cam_get_view_matrix(cam));
}