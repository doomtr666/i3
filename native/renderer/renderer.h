#pragma once

#include "render_graph.h"
#include "scene.h"

#define I3_RENDERER_LOGGER_NAME "renderer"

typedef struct i3_renderer_o i3_renderer_o;

// render target image and view
typedef struct i3_render_target_t
{
    i3_rbk_image_i* image;            // render target image
    i3_rbk_image_view_i* image_view;  // view of the render target image
} i3_render_target_t;

struct i3_renderer_i
{
    i3_renderer_o* self;

    // create a model
    i3_model_i* (*create_model)(i3_renderer_o* self, i3_rbk_cmd_buffer_i* cmb_buffer, i3_content_i* model_content);

    // create a scene
    i3_scene_i* (*create_scene)(i3_renderer_o* self);

    // set the current scene
    void (*set_scene)(i3_renderer_o* self, i3_scene_i* scene);

    // get the current scene
    i3_scene_i* (*get_scene)(i3_renderer_o* self);

    // manage render graphs
    i3_render_graph_builder_i* (*create_graph_builder)(i3_renderer_o* self);
    void (*set_render_graph)(i3_renderer_o* self, i3_render_graph_i* graph);

    // create render target, this will create an image and a view for the render target
    // if the image and view already exist, it will be recreated
    void (*create_render_target)(i3_renderer_o* self,
                                 i3_render_target_t* target,
                                 i3_rbk_image_desc_t* image_desc,
                                 i3_rbk_image_view_desc_t* view_desc);

    void (*destroy_render_target)(i3_renderer_o* self, i3_render_target_t* target);

    void (*render)(i3_renderer_o* self, i3_game_time_t* game_time);

    void (*destroy)(i3_renderer_o* self);
};

i3_renderer_i* i3_renderer_create(i3_render_backend_i* backend,
                                  i3_render_window_i* window,
                                  i3_content_store_i* content_store);
