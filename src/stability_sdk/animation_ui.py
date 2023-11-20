import glob
import json
import locale
import os
import param
import shutil
import traceback

from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, List, Optional

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Failed to import animation UI requirements. To use the animation UI, install the dependencies with:\n" 
        "   pip install --upgrade stability_sdk[anim_ui]"
    )

from .api import (
    ClassifierException, 
    Context,
    OutOfCreditsException,
)
from .animation import (
    AnimationArgs,
    Animator,
    AnimationSettings,
    BasicSettings,
    CameraSettings,
    CoherenceSettings,
    ColorSettings,
    DepthSettings,
    InpaintingSettings,
    Rendering3dSettings,
    VideoInputSettings,
    VideoOutputSettings,
    interpolate_frames
)
from .utils import (
    create_video_from_frames,
    extract_frames_from_video,
    interpolate_mode_from_string
)


DATA_VERSION = "0.1"
DATA_GENERATOR = "stability_sdk.animation_ui"

PRESETS = {
    "Default": {},
    "3D warp rotate": {
        "animation_mode": "3D warp", "rotation_y":"0:(0.4)", "translation_x":"0:(-1.2)", "depth_model_weight":1.0,
        "animation_prompts": "{\n0:\"a flower vase on a table\"\n}"
    },
    "3D warp zoom": {
        "animation_mode":"3D warp", "diffusion_cadence_curve":"0:(4)", "noise_scale_curve":"0:(1.04)", 
        "strength_curve":"0:(0.7)", "translation_z":"0:(1.0)",
    },
    "3D render rotate": {
        "animation_mode": "3D render", "depth_model_weight":1.0,
        "translation_x":"0:(-3.5)", "rotation_y":"0:(1.7)", "translation_z":"0:(-0.5)",
        "diffusion_cadence_curve":"0:(1)", "strength_curve":"0:(0.96)", "noise_scale_curve":"0:(1.01)",
        "mask_min_value":"0:(0.35)", "use_inpainting_model":False, "preset": "anime",
        "animation_prompts": "{\n0:\"beautiful portrait of a ninja in a sunflower field\"\n}"
    },
    "3D render explore": {
        "animation_mode": "3D render", "translation_z":"0:(10)", "translation_x":"0:(2), 20:(-2), 40:(2)",
        "rotation_y":"0:(0), 10:(1.5), 30:(-2), 50: (3)", "rotation_x":"0:(0.4)",
        "diffusion_cadence_curve":"0:(1)", "strength_curve":"0:(0.98)",
        "noise_scale_curve":"0:(1.01)", "depth_model_weight":1.0,
        "mask_min_value":"0:(0.1)", "use_inpainting_model":False, "preset":"3d-model",
        "animation_prompts": "{\n0:\"Phantasmagoric carnival, carnival attractions shifting and changing, bizarre surreal circus\"\n}"
    },
    "Prompt interpolate": {
        "animation_mode":"2D", "interpolate_prompts":True, "locked_seed":True, "max_frames":24, 
        "strength_curve":"0:(0)", "diffusion_cadence_curve":"0:(4)", "cadence_interp":"film",
        "clip_guidance":"None", "animation_prompts": "{\n0:\"a photo of a cute cat\",\n24:\"a photo of a cute dog\"\n}"
    },
    "Translate and inpaint": {
        "animation_mode":"2D", "inpaint_border":True, "use_inpainting_model":False, "translation_x":"0:(-20)", 
        "diffusion_cadence_curve":"0:(3)", "strength_curve":"0:(0.85)", "noise_scale_curve":"0:(1.01)", "border":"reflect",
        "animation_prompts": "{\n0:\"Mystical pumpkin field landscapes on starry Halloween night, pop surrealism art\"\n}"
    },
    "Outpaint": {
        "animation_mode":"2D", "diffusion_cadence_curve":"0:(16)", "cadence_spans":True, "use_inpainting_model":True,
        "strength_curve":"0:(1)", "reverse":True, "preset": "fantasy-art", "inpaint_border":True, "zoom":"0:(0.95)", 
        "animation_prompts": "{\n0:\"an ancient and magical portal, in a fantasy corridor\"\n}"
    },
    "Video Stylize": {
        "animation_mode":"Video Input", "model":"stable-diffusion-depth-v2-0", "locked_seed":True, 
        "strength_curve":"0:(0.22)", "clip_guidance":"None", "video_mix_in_curve":"0:(1.0)", "video_flow_warp":True,
    },
}

class Project():
    def __init__(self, title, settings={}) -> None:
        self.folder = title.replace("/", "_").replace("\\", "_").replace(":", "")
        self.settings = settings
        self.title = title
    
    @classmethod
    def list_projects(cls) -> List["Project"]:
        projects = []
        for path in os.listdir(outputs_path):
            directory = os.path.join(outputs_path, path)
            if not os.path.isdir(directory):
                continue

            json_files = glob.glob(os.path.join(directory, '*.json'))
            json_files = sorted(json_files, key=lambda x: os.stat(x).st_mtime)
            if not json_files:
                continue

            filename = os.path.basename(json_files[-1])
            if not '(' in filename:
                continue

            project = cls(filename[:filename.rfind('(')-1].strip())
            try:
                project.settings = json.load(open(os.path.join(directory, filename), 'r'))
            except:
                continue
            projects.append(project)
        return projects


context = None
outputs_path = None

args_generation = BasicSettings()
args_animation = AnimationSettings()
args_camera = CameraSettings()
args_coherence = CoherenceSettings()
args_color = ColorSettings()
args_depth = DepthSettings()
args_render_3d = Rendering3dSettings()
args_inpaint = InpaintingSettings()
args_vid_in = VideoInputSettings()
args_vid_out = VideoOutputSettings()
arg_objs = (
    args_generation,
    args_animation,
    args_camera,
    args_coherence,
    args_color,
    args_depth,
    args_render_3d,
    args_inpaint,
    args_vid_in,
    args_vid_out,
)

animation_prompts = "{\n0: \"\"\n}"
negative_prompt = "blurry, low resolution"
negative_prompt_weight = -1.0

controls: Dict[str, gr.components.Component] = {}
header = gr.HTML("")
interrupt = False
last_interp_factor = None
last_interp_mode = None
last_project_settings_path = None
last_upscale = None
projects: List[Project] = []
project: Optional[Project] = None
resume_checkbox = gr.Checkbox(label="Resume", value=False, interactive=True)
resume_from_number = gr.Number(label="Resume from frame", value=-1, interactive=True, precision=0,
                               info="Positive frame number to resume from, or -1 to resume from the last")

project_create_button = gr.Button("Create")
project_data_log = gr.Textbox(label="Status", visible=False)
project_load_button = gr.Button("Load")
project_new_title = gr.Text(label="Name", value="My amazing animation", interactive=True)
project_preset_dropdown = gr.Dropdown(label="Preset", choices=list(PRESETS.keys()), value=list(PRESETS.keys())[0], interactive=True)
project_row_create = None
project_row_import = None
project_row_load = None
projects_dropdown = gr.Dropdown([p.title for p in projects], label="Project", visible=True, interactive=True)

project_import_button = gr.Button("Import")
project_import_file = gr.File(label="Project file", file_types=[".json", ".txt"], type="binary")
project_import_title = gr.Text(label="Name", value="Imported project", interactive=True)


def accordion_for_color(args: ColorSettings):
    p = args.param
    with gr.Accordion("Color", open=False):
        controls["color_coherence"] = gr.Dropdown(label="Color coherence", choices=p.color_coherence.objects, value=p.color_coherence.default, interactive=True)
        with gr.Row():
            controls["brightness_curve"] = gr.Text(label="Brightness curve", value=p.brightness_curve.default, interactive=True)
            controls["contrast_curve"] = gr.Text(label="Contrast curve", value=p.contrast_curve.default, interactive=True)
        with gr.Row():
            controls["hue_curve"] = gr.Text(label="Hue curve", value=p.hue_curve.default, interactive=True)
            controls["saturation_curve"] = gr.Text(label="Saturation curve", value=p.saturation_curve.default, interactive=True)
            controls["lightness_curve"] = gr.Text(label="Lightness curve", value=p.lightness_curve.default, interactive=True)
        controls["color_match_animate"] = gr.Checkbox(label="Animated color match", value=p.color_match_animate.default, interactive=True)

def accordion_from_args(name: str, args: param.Parameterized, exclude: List[str]=[], open=False):
    with gr.Accordion(name, open=open):
        ui_from_args(args, exclude)

def args_reset_to_defaults():
    for args in arg_objs:
        for k, v in args.param.objects().items():
            if k == "name":
                continue
            setattr(args, k, v.default)

def args_to_controls(data: Optional[dict]=None) -> dict:    
    # go through all the parameters and load their settings from the data
    global animation_prompts, negative_prompt
    if data:
        for arg in arg_objs:
            for k, v in arg.param.objects().items():
                if k != "name" and k in data:
                    arg.param.set_param(k, data[k])
        if "animation_prompts" in data:
            animation_prompts = data["animation_prompts"]
        if "negative_prompt" in data:
            negative_prompt = data["negative_prompt"]

    returns = {}
    returns[controls['animation_prompts']] = gr.update(value=animation_prompts)
    returns[controls['negative_prompt']] = gr.update(value=negative_prompt)

    for args in arg_objs:
        for k, v in args.param.objects().items():
            if k in controls:
                c = controls[k]
                returns[c] = gr.update(value=getattr(args, k))

    return returns

def ensure_api_context():
    if context is None:
        raise gr.Error("Not connected to Stability API")

def format_header_html() -> str:
    try:
        balance, profile_picture = context.get_user_info()
    except:
        return ""
    formatted_number = locale.format_string("%d", balance, grouping=True)
    return f"""
        <div class="flex flex-row items-center" style="display:flex; justify-content: space-between; margin-top: 8px;">
            <div>Stable Animation UI</div>
            <div class="flex cursor-pointer flex-row items-center gap-1" style="display:flex; gap: 0.25rem; justify-content: flex-end;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4">
                    <circle cx="8" cy="8" r="6"></circle>
                    <path d="M18.09 10.37A6 6 0 1 1 10.34 18"></path>
                    <path d="M7 6h1v4"></path>
                    <path d="m16.71 13.88.7.71-2.82 2.82"></path>
                </svg>
                {formatted_number}
                <div style="width:28px; height:28px; overflow:hidden; border-radius:50%;">
                    <img alt="user avatar" src="{profile_picture}" class="MuiAvatar-img css-1hy9t21">
                </div>
            </div>
        </div>
    """

def get_default_project():
    data = OrderedDict(AnimationArgs().param.values())
    data.update({
        "version": DATA_VERSION,
        "generator": DATA_GENERATOR
    })
    return data

def post_process_tab():
    with gr.Row():
        with gr.Column():
            with gr.Row(visible=False):
                use_video_instead = gr.Checkbox(label="Postprocess a video instead", value=False, interactive=True)
                video_to_postprocess = gr.Text(label="Videofile to postprocess", value="", interactive=True)
            fps = gr.Number(label="Output FPS", value=24, interactive=True, precision=0)
            reverse = gr.Checkbox(label="Reverse", value=False, interactive=True)
            with gr.Row():
                frame_interp_mode = gr.Dropdown(label="Frame interpolation mode", choices=['None', 'film', 'rife'], value='None', interactive=True)       
                frame_interp_factor = gr.Dropdown(label="Frame interpolation factor", choices=[2, 4, 8], value=2, interactive=True)
            with gr.Row():
                upscale = gr.Checkbox(label="Upscale 2X", value=False, interactive=True)
        with gr.Column():
            image_out = gr.Image(label="image", visible=True)
            video_out = gr.Video(label="video", visible=False)
            process_button = gr.Button("Process")
            stop_button = gr.Button("Stop", visible=False)
            status = gr.Textbox(lines=3, visible=False)

    def postprocess_video(fps: int, reverse: bool, interp_mode: str, interp_factor: int, upscale: bool,
                          use_video_instead: bool, video_to_postprocess: str):
        global interrupt, last_interp_factor, last_interp_mode, last_upscale
        interrupt = False
        if not use_video_instead and last_project_settings_path is None:
            raise gr.Error("Please render an animation first or specify a videofile to postprocess")
        if use_video_instead and not os.path.exists(video_to_postprocess):
            raise gr.Error("Videofile does not exist")

        yield {
            image_out: gr.update(visible=True, label="", value=None),
            video_out: gr.update(visible=False),
            process_button: gr.update(visible=False),
            stop_button: gr.update(visible=True),
            status: gr.update(visible=False),
        }

        error, output_video = None, None
        try:
            outdir = os.path.dirname(last_project_settings_path) \
                if not use_video_instead \
                else extract_frames_from_video(video_to_postprocess)
            suffix = ""

            can_skip_upscale = last_upscale == upscale
            can_skip_interp = can_skip_upscale and last_interp_factor == interp_factor and last_interp_mode == interp_mode

            if upscale:
                suffix += "_x2"
                upscale_dir = os.path.join(outdir, "upscale") 
                os.makedirs(upscale_dir, exist_ok=True)
                frame_paths = sorted(glob.glob(os.path.join(outdir, "frame_*.png")))
                num_frames = len(frame_paths)
                if not can_skip_upscale:
                    remove_frames_from_path(upscale_dir)
                    for frame_idx in tqdm(range(num_frames)):
                        frame = Image.open(frame_paths[frame_idx])
                        frame = context.upscale(frame)
                        frame.save(os.path.join(upscale_dir, os.path.basename(frame_paths[frame_idx])))
                        yield {
                            header: gr.update(value=format_header_html()) if frame_idx % 12 == 0 else gr.update(),
                            image_out: gr.update(value=frame, label=f"upscale {frame_idx}/{num_frames}", visible=True),
                        }
                        if interrupt:
                            break
                    last_upscale = upscale
                outdir = upscale_dir

            if interp_mode != 'None':
                suffix += f"_{interp_mode}{interp_factor}"
                interp_dir = os.path.join(outdir, "interpolate")
                interp_mode = interpolate_mode_from_string(interp_mode)
                if not can_skip_interp:
                    remove_frames_from_path(interp_dir)
                    num_frames = interp_factor * len(glob.glob(os.path.join(outdir, "frame_*.png")))
                    for frame_idx, frame in enumerate(tqdm(interpolate_frames(context, outdir, interp_dir, interp_mode, interp_factor), total=num_frames)):
                        yield {
                            header: gr.update(value=format_header_html()) if frame_idx % 12 == 0 else gr.update(),
                            image_out: gr.update(value=frame, label=f"interpolate {frame_idx}/{num_frames}", visible=True),
                        }
                        if interrupt:
                            break
                    last_interp_mode, last_interp_factor = interp_mode, interp_factor
                outdir = interp_dir

            if not use_video_instead:
                output_video = last_project_settings_path.replace(".json", f"{suffix}.mp4")
            else:
                _, video_ext = os.path.splitext(video_to_postprocess)
                output_video = video_to_postprocess.replace(video_ext, f"{suffix}.mp4")

            yield { status: gr.update(label="Status", value="Compiling frames to MP4...", visible=True) }
            create_video_from_frames(outdir, output_video, fps=fps, reverse=reverse)
        except Exception as e:
            traceback.print_exc()
            error = f"Post-processing terminated early due to exception: {e}"

        yield {
            header: gr.update(value=format_header_html()),
            image_out: gr.update(visible=False),
            video_out: gr.update(value=output_video, visible=True),
            process_button: gr.update(visible=True),
            stop_button: gr.update(visible=False),
            status: gr.update(label="Error", value=error, visible=bool(error))
        }

    process_button.click(
        postprocess_video, 
        inputs=[fps, reverse, frame_interp_mode, frame_interp_factor, upscale, use_video_instead, video_to_postprocess], 
        outputs=[header, image_out, video_out, process_button, stop_button, status]
    )    

    def stop():
        global interrupt
        interrupt = True
        return { status: gr.update(label="Status", value="Stopping...", visible=True)}
    stop_button.click(stop, outputs=[status])


def project_create(title, preset):
    ensure_api_context()
    global project, projects
    titles = [p.title for p in projects]
    if title in titles:
        raise gr.Error(f"Project with title '{title}' already exists")
    project = Project(title, get_default_project())
    projects.append(project)
    projects = sorted(projects, key=lambda p: p.title)

    # grab each setting from the preset and add to settings
    for k, v in PRESETS[preset].items():
        project.settings[k] = v

    log = f"Created project '{title}'"

    args_reset_to_defaults()
    returns = args_to_controls(project.settings)
    returns[project_data_log] = gr.update(value=log, visible=True)
    returns[projects_dropdown] = gr.update(choices=[p.title for p in projects], visible=True, value=title)
    returns[project_row_load] = gr.update(visible=len(projects) > 0)
    return returns

def project_import(title, file):
    ensure_api_context()
    global project, projects
    titles = [p.title for p in projects]
    if title in titles:
        raise gr.Error(f"Project with title '{title}' already exists")

    # read json from file
    try:
        settings = json.loads(file.decode('utf-8'))
    except Exception as e:
        raise gr.Error(f"Failed to read settings from file: {e}")

    project = Project(title, settings)
    projects.append(project)
    projects = sorted(projects, key=lambda p: p.title)

    log = f"Imported project '{title}'"

    args_reset_to_defaults()
    returns = args_to_controls(project.settings)
    returns[project_data_log] = gr.update(value=log, visible=True)
    returns[projects_dropdown] = gr.update(choices=[p.title for p in projects], visible=True, value=title)
    returns[project_row_load] = gr.update(visible=len(projects) > 0)
    return returns

def project_load(title: str):
    ensure_api_context()
    global project
    project = next(p for p in projects if p.title == title)
    data = project.settings

    log = f"Loaded project '{title}'"

    # filter project file to latest version
    if "animation_mode" in data and data["animation_mode"] == "3D":
        data["animation_mode"] = "3D warp"
    if "midas_weight" in data:
        data["depth_model_weight"] = data["midas_weight"]
        del data["midas_weight"]

    # update the ui controls
    returns = args_to_controls(data)
    returns[project_data_log] = gr.update(value=log, visible=True)
    return returns

def project_tab():
    global project_row_create, project_row_import, project_row_load

    button_load_projects = gr.Button("Load Projects")
    with gr.Accordion("Load a project", open=True, visible=False) as projects_row_:
        project_row_load = projects_row_
        with gr.Row():
            projects_dropdown.render()
            with gr.Column():
                project_load_button.render()
                with gr.Row():
                    delete_btn = gr.Button("Delete")
                    confirm_btn = gr.Button("Confirm delete", variant="stop", visible=False)
                    cancel_btn = gr.Button("Cancel", visible=False)                
                delete_btn.click(lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [delete_btn, confirm_btn, cancel_btn])
                cancel_btn.click(lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [delete_btn, confirm_btn, cancel_btn])

    with gr.Accordion("Create a new project", open=True, visible=False) as project_row_create_:
        project_row_create = project_row_create_
        with gr.Column():
            with gr.Row():
                project_new_title.render()
                project_preset_dropdown.render()
            with gr.Column():
                project_create_button.render()

    with gr.Accordion("Import a project file", open=False, visible=False) as project_row_import_:
        project_row_import = project_row_import_
        with gr.Column():
            with gr.Row():
                project_import_title.render()
                project_import_file.render()
            with gr.Column():
                project_import_button.render()

    project_data_log.render()

    def delete_project(title: str):
        ensure_api_context()
        global project, projects

        project = next(p for p in projects if p.title == title)
        project_path = os.path.join(outputs_path, project.folder)
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

        projects.remove(project)
        project = None

        log = f"Deleted project \"{title}\" at \"{project_path}\""
        return {
            projects_dropdown: gr.update(choices=[p.title for p in projects], visible=True),
            project_row_load: gr.update(visible=len(projects) > 0),
            project_data_log: gr.update(value=log, visible=True),
            delete_btn: gr.update(visible=True), 
            confirm_btn: gr.update(visible=False), 
            cancel_btn: gr.update(visible=False)
        }

    def load_projects():
        ensure_api_context()
        global projects
        projects = Project.list_projects()
        return {
            button_load_projects: gr.update(visible=False),
            projects_dropdown: gr.update(choices=[p.title for p in projects], visible=True),
            project_row_create: gr.update(visible=True),
            project_row_import: gr.update(visible=True),
            project_row_load: gr.update(visible=len(projects) > 0),
            header: gr.update(value=format_header_html())
        }

    button_load_projects.click(load_projects, outputs=[button_load_projects, projects_dropdown, project_row_create, project_row_import, project_row_load, header])
    confirm_btn.click(delete_project, inputs=projects_dropdown, outputs=[projects_dropdown, project_row_load, project_data_log, delete_btn, confirm_btn, cancel_btn])

def remove_frames_from_path(path: str, leave_first: Optional[int]=None):
    if os.path.isdir(path):
        frames = sorted(glob.glob(os.path.join(path, "frame_*.png")))
        if leave_first:
            frames = frames[leave_first:]
        for f in frames:
            os.remove(f)

def render_tab():
    with gr.Row():
        with gr.Column():
            ui_layout_tabs()
        with gr.Column():
            image_out = gr.Image(label="image", visible=True)
            video_out = gr.Video(label="video", visible=False)
            button = gr.Button("Render")
            button_stop = gr.Button("Stop", visible=False)
            status = gr.Textbox(lines=3, visible=False)

    def render(resume: bool, resume_from: int, *render_args):
        global interrupt, last_interp_factor, last_interp_mode, last_project_settings_path, last_upscale, project
        interrupt = False

        if not project:
            raise gr.Error("No project active!")
        
        # create local folder for the project
        outdir = os.path.join(outputs_path, project.folder)
        os.makedirs(outdir, exist_ok=True)

        # each render gets a unique run index
        run_index = 0
        while True:
            project_settings_path = os.path.join(outdir, f"{project.folder} ({run_index}).json")
            if not os.path.exists(project_settings_path):
                break
            run_index += 1

        # gather up all the settings from sub-objects
        args_d = {k: v for k, v in zip(controls.keys(), render_args)}
        animation_prompts, negative_prompt = args_d['animation_prompts'], args_d['negative_prompt']
        del args_d['animation_prompts'], args_d['negative_prompt']
        args = AnimationArgs(**args_d)

        if args.animation_mode == "Video Input" and not args.video_init_path:
            raise gr.Error("No video input file selected!")

        # convert animation_prompts from string (JSON or python) to dict
        try:
            prompts = json.loads(animation_prompts)
        except json.JSONDecodeError:
            try:
                prompts = eval(animation_prompts)
            except Exception as e:
                raise gr.Error("Invalid JSON or Python code for animation_prompts!")
        prompts = {int(k): v for k, v in prompts.items()}

        # save settings to a dict
        save_dict = OrderedDict()
        save_dict['version'] = DATA_VERSION
        save_dict['generator'] = DATA_GENERATOR
        save_dict.update(args.param.values())
        save_dict['animation_prompts'] = animation_prompts
        save_dict['negative_prompt'] = negative_prompt
        project.settings = save_dict
        with open(project_settings_path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=4)

        # initial yield to switch render button to stop button
        yield {
            button: gr.update(visible=False),
            button_stop: gr.update(visible=True),
            image_out: gr.update(visible=True, label="", value=None),
            video_out: gr.update(visible=False),
            status: gr.update(visible=False),
        }

        # delete frames from previous animation
        if resume:
            if resume_from > 0:
                remove_frames_from_path(outdir, resume_from)
            elif resume_from == 0 or resume_from < -1:
                raise gr.Error("Frame number to resume from must be positive, or -1 to resume from the last frame")
        else:
            remove_frames_from_path(outdir)

        frame_idx, error = 0, None
        try:
            animator = Animator(
                api_context=context,
                animation_prompts=prompts,
                args=args,
                out_dir=outdir,
                negative_prompt=negative_prompt,
                negative_prompt_weight=negative_prompt_weight,
                resume=resume,
            )
            for frame_idx, frame in enumerate(tqdm(animator.render(), initial=animator.start_frame_idx, total=args.max_frames), start=animator.start_frame_idx):
                if interrupt:
                    break

                yield {
                    image_out: gr.update(value=frame, label=f"frame {frame_idx}/{args.max_frames}", visible=True),
                    header: gr.update(value=format_header_html()) if frame_idx % 12 == 0 else gr.update(),
                }
        except ClassifierException as e:
            error = "Animation terminated early due to NSFW classifier."
            if e.prompt is not None:
                error += "\nPlease revise your prompt: " + e.prompt
        except OutOfCreditsException as e:
            error = f"Animation terminated early, out of credits.\n{e.details}"
        except Exception as e:
            traceback.print_exc()
            error = f"Animation terminated early due to exception: {e}"

        if frame_idx:
            last_project_settings_path = project_settings_path
            last_interp_factor, last_interp_mode, last_upscale = None, None, None
            output_video = project_settings_path.replace(".json", ".mp4")
            yield {
                status: gr.update(label="Status", value="Compiling frames to MP4...", visible=True),
            }
            try:
                create_video_from_frames(outdir, output_video, fps=args.fps, reverse=args.reverse)
            except RuntimeError as e:
                error = f"Error creating video: {e}"
                output_video = None
        else:
            output_video = None
        yield {
            button: gr.update(visible=True),
            button_stop: gr.update(visible=False),
            image_out: gr.update(visible=False),
            video_out: gr.update(value=output_video, visible=True),
            header: gr.update(value=format_header_html()),
            status: gr.update(label="Error", value=error, visible=bool(error)),
        }

    button.click(
        render,
        inputs=[resume_checkbox, resume_from_number] + list(controls.values()),
        outputs=[button, button_stop, image_out, video_out, header, status]
    )

    # stop animation in progress 
    def stop():
        global interrupt
        interrupt = True
        return { status: gr.update(label="Status", value="Stopping...", visible=True) }
    button_stop.click(stop, outputs=[status])

def ui_for_animation_settings(args: AnimationSettings):
    with gr.Row():
        controls["steps_strength_adj"] = gr.Checkbox(label="Steps strength adj", value=args.param.steps_strength_adj.default, interactive=True)
        controls["interpolate_prompts"] = gr.Checkbox(label="Interpolate prompts", value=args.param.interpolate_prompts.default, interactive=True)
        controls["locked_seed"] = gr.Checkbox(label="Locked seed", value=args.param.locked_seed.default, interactive=True)
    controls["noise_add_curve"] = gr.Text(label="Noise add curve", value=args.param.noise_add_curve.default, interactive=True)
    controls["noise_scale_curve"] = gr.Text(label="Noise scale curve", value=args.param.noise_scale_curve.default, interactive=True)
    controls["strength_curve"] = gr.Text(label="Previous frame strength curve", value=args.param.strength_curve.default, interactive=True)
    controls["steps_curve"] = gr.Text(label="Steps curve", value=args.param.steps_curve.default, interactive=True)

def ui_for_generation(args: AnimationSettings):
    p = args.param
    with gr.Row():
        controls["width"] = gr.Number(label="Width", value=p.width.default, interactive=True, precision=0)
        controls["height"] = gr.Number(label="Height", value=p.height.default, interactive=True, precision=0)
    with gr.Row():
        controls["model"] = gr.Dropdown(label="Model", choices=p.model.objects, value=p.model.default, interactive=True)
        controls["custom_model"] = gr.Text(label="Custom model", value=p.custom_model.default, interactive=True)
    with gr.Row():
        controls["preset"] = gr.Dropdown(label="Style preset", choices=p.preset.objects, value=p.preset.default, interactive=True)
    with gr.Row():
        controls["sampler"] = gr.Dropdown(label="Sampler", choices=p.sampler.objects, value=p.sampler.default, interactive=True)
        controls["seed"] = gr.Number(label="Seed", value=p.seed.default, interactive=True, precision=0)
        controls["cfg_scale"] = gr.Number(label="Guidance scale", value=p.cfg_scale.default, interactive=True)
        controls["clip_guidance"] = gr.Dropdown(label="CLIP guidance", choices=p.clip_guidance.objects, value=p.clip_guidance.default, interactive=True)

def ui_for_init_and_mask(args_generation):
    p = args_generation.param
    with gr.Row():
        controls["init_image"] = gr.Text(label="Init image", value=p.init_image.default, interactive=True)
        controls["init_sizing"] = gr.Dropdown(label="Init sizing", choices=p.init_sizing.objects, value=p.init_sizing.default, interactive=True)
    with gr.Row():
        controls["mask_path"] = gr.Text(label="Mask path", value=p.mask_path.default, interactive=True)
        controls["mask_invert"] = gr.Checkbox(label="Mask invert", value=p.mask_invert.default, interactive=True)

def ui_for_video_output(args: VideoOutputSettings):
    p = args.param
    controls["fps"] = gr.Number(label="FPS", value=p.fps.default, interactive=True, precision=0)
    controls["reverse"] = gr.Checkbox(label="Reverse", value=p.reverse.default, interactive=True)

def ui_from_args(args: param.Parameterized, exclude: List[str]=[]):
    for k, v in args.param.objects().items():
        if k == "name" or k in exclude:
            continue
        if isinstance(v, param.Boolean):
            t = gr.Checkbox(label=v.label, value=v.default, interactive=True)
        elif isinstance(v, param.Integer):
            t = gr.Number(label=v.label, value=v.default, interactive=True, precision=0)
        elif isinstance(v, param.Number):
            t = gr.Number(label=v.label, value=v.default, interactive=True)
        elif isinstance(v, param.Selector):
            t = gr.Dropdown(label=v.label, choices=v.objects, value=v.default, interactive=True)
        elif isinstance(v, param.String):
            t = gr.Text(label=v.label, value=v.default, interactive=True)
        else:
            raise Exception(f"Unknown parameter type {v} for param {k}")
        controls[k] = t

def ui_layout_tabs():
    with gr.Tab("Prompts"):
        with gr.Row():
            controls['animation_prompts'] = gr.TextArea(label="Animation prompts", max_lines=8, value=animation_prompts, interactive=True)
        with gr.Row():
            controls['negative_prompt'] = gr.Textbox(label="Negative prompt", max_lines=1, value=negative_prompt, interactive=True)
    with gr.Tab("Config"):
        with gr.Row():
            args = args_animation
            controls["animation_mode"] = gr.Dropdown(label="Animation mode", choices=args.param.animation_mode.objects, value=args.param.animation_mode.default, interactive=True)
            controls["max_frames"] = gr.Number(label="Max frames", value=args.param.max_frames.default, interactive=True, precision=0)
            controls["border"] = gr.Dropdown(label="Border", choices=args.param.border.objects, value=args.param.border.default, interactive=True)
        ui_for_generation(args_generation)
        ui_for_animation_settings(args_animation)
        accordion_from_args("Coherence", args_coherence, open=False)
        accordion_for_color(args_color)
        accordion_from_args("Depth", args_depth, exclude=["near_plane", "far_plane"], open=False)
        accordion_from_args("3D render", args_render_3d, open=False)
        accordion_from_args("Inpainting", args_inpaint, open=False)
    with gr.Tab("Input"):
        with gr.Row():
            resume_checkbox.render()
            resume_from_number.render()
        ui_for_init_and_mask(args_generation)
        with gr.Column():
            p = args_vid_in.param
            with gr.Row():
                controls["video_init_path"] = gr.Text(label="Video init path", value=p.video_init_path.default, interactive=True)
            with gr.Row():
                controls["video_mix_in_curve"] = gr.Text(label="Mix in curve", value=p.video_mix_in_curve.default, interactive=True)
                controls["extract_nth_frame"] = gr.Number(label="Extract nth frame", value=p.extract_nth_frame.default, interactive=True, precision=0)
                controls["video_flow_warp"] = gr.Checkbox(label="Flow warp", value=p.video_flow_warp.default, interactive=True)

    with gr.Tab("Camera"):
        p = args_camera.param
        gr.Markdown("2D Camera")
        controls["angle"] = gr.Text(label="Angle", value=p.angle.default, interactive=True)
        controls["zoom"] = gr.Text(label="Zoom", value=p.zoom.default, interactive=True)

        gr.Markdown("2D and 3D Camera translation")
        controls["translation_x"] = gr.Text(label="Translation X", value=p.translation_x.default, interactive=True)
        controls["translation_y"] = gr.Text(label="Translation Y", value=p.translation_y.default, interactive=True)
        controls["translation_z"] = gr.Text(label="Translation Z", value=p.translation_z.default, interactive=True)

        gr.Markdown("3D Camera rotation")
        controls["rotation_x"] = gr.Text(label="Rotation X", value=p.rotation_x.default, interactive=True)
        controls["rotation_y"] = gr.Text(label="Rotation Y", value=p.rotation_y.default, interactive=True)
        controls["rotation_z"] = gr.Text(label="Rotation Z", value=p.rotation_z.default, interactive=True)

    with gr.Tab("Output"):
        ui_for_video_output(args_vid_out)


def create_ui(api_context: Context, outputs_root_path: str):
    global context, outputs_path, projects
    context, outputs_path = api_context, outputs_root_path

    locale.setlocale(locale.LC_ALL, '')

    with gr.Blocks() as ui:
        header.render()

        with gr.Tab("Project"):
            project_tab()

        with gr.Tab("Render"):
            render_tab()

        with gr.Tab("Post-process"):
            post_process_tab()

        load_project_outputs = [project_data_log]
        load_project_outputs.extend(controls.values())
        project_load_button.click(project_load, inputs=projects_dropdown, outputs=load_project_outputs)

        create_project_outputs = [project_data_log, projects_dropdown, project_row_load]
        create_project_outputs.extend(controls.values())
        project_create_button.click(project_create, inputs=[project_new_title, project_preset_dropdown], outputs=create_project_outputs)
        project_import_button.click(project_import, inputs=[project_import_title, project_import_file], outputs=create_project_outputs)

    return ui
