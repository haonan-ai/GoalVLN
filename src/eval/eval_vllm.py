import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import re
from tqdm import tqdm
import cv2
import imageio
from habitat.utils.visualizations import maps
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector, quaternion_to_list
import random
from vllm import SamplingParams
import argparse, habitat
from habitat_extensions import measures, task
from habitat_baselines.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from PIL import Image, ImageFont, ImageDraw
from qwen_vl_utils import process_vision_info
import multiprocessing as mp
import queue
import time, math
from openai import OpenAI
from vllm.multimodal.utils import encode_image_base64


SYSTEM_PROMPT = "You are a helpful assistant."

def seed_all():
    np.random.seed(41)
    random.seed(41)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_episode_ids(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def filter_dataset_by_episode_ids(dataset, episode_ids_file):
    target_ids = load_episode_ids(episode_ids_file)
    filtered = [
        ep for ep in dataset.episodes if str(ep.episode_id) in target_ids
    ]
    if not filtered:
        raise ValueError(f"No episodes matched IDs in {episode_ids_file}")
    found_ids = {str(ep.episode_id) for ep in filtered}
    missing = target_ids - found_ids
    if missing:
        print(f"Warning: {len(missing)} IDs from file not found in dataset")
    dataset.episodes = filtered
    return dataset


def agent_state_to_pose(agent_state):
    return {
        "position": np.array(agent_state.position, dtype=np.float64).tolist(),
        "rotation": quaternion_to_list(agent_state.rotation),
    }


def format_goal_hint(current_pose, goal_pose):
    current_position = np.array(current_pose["position"], dtype=np.float64)
    goal_position = np.array(goal_pose["position"], dtype=np.float64)
    current_rotation = quaternion_from_coeff(current_pose["rotation"])

    direction_vector = goal_position - current_position
    direction_vector_agent = quaternion_rotate_vector(
        current_rotation.inverse(), direction_vector
    )

    x_forward = -direction_vector_agent[2]
    y_lateral = -direction_vector_agent[0]
    x_abs = abs(x_forward)
    y_abs = abs(y_lateral)
    front_or_back = "front" if x_forward >= 0 else "back"
    left_or_right = "left" if y_lateral >= 0 else "right"

    return (
        f"Goal hint: Relative to the robot, the goal is {x_abs:.1f} m {front_or_back} "
        f"and {y_abs:.1f} m to the {left_or_right}."
    )


def evaluate_agent(result_queue, api_key, base_url, config, dataset, result_path, num_generations,
                    forward_distance, turn_angle, max_action_history, resolution_ratio,
                    use_goal_hint=False, save_gif=True, force_reeval=False) -> None:
 
    env = Env(config.habitat, dataset)

    agent = NaVIDA_Agent(
        api_key, 
        base_url, 
        result_path, 
        forward_distance, 
        turn_angle, 
        max_action_history, 
        resolution_ratio, 
        num_generations,
        use_goal_hint=use_goal_hint,
        save_gif=save_gif)

    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = 25
    EARLY_STOP_STEPS = 400

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success","ndtw"}

    count = 0
    
    for _ in range(num_episodes):
        episode_start_time = time.time()

        obs = env.reset()
        iter_step = 0
        agent.reset()

        t_dict = {
            "t_episode": 0,
        }

        continuse_rotation_count = 0
        last_dtg = 999
        stats_path = os.path.join(result_path, "log", f"stats_{env.current_episode.episode_id}.json")
        if not force_reeval and os.path.exists(stats_path):
            result_queue.put({"t_episode": 0})
            continue
        while not env.episode_over:
            
            info = env.get_metrics()
            
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            
            action = agent.act(obs, info, env.current_episode.episode_id, env)

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}

            
            iter_step+=1
            obs = env.step(action)
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1

        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)

        agent.save_episode_video()
        
        t_dict["t_episode"] = time.time() - episode_start_time
        result_queue.put(t_dict)

class NaVIDA_Agent(Agent):
    def __init__(self, api_key, base_url, result_path, forward_distance, 
                    turn_angle, max_action_history, resolution_ratio, num_generations = 1,
                    save_gif=True, use_goal_hint=False):
        
        print("Initialize NaVIDA")
        
        self.result_path = result_path
        self.save_gif = save_gif
        self.require_map = save_gif
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        self.use_goal_hint = use_goal_hint
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        if self.save_gif:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id
        
        self.sampling_params = SamplingParams(
            n = 1,
            temperature=0.2,
            max_tokens=512, # i.e. max_completion_tokens
        )

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. "\
            "You have been given a video of historical observations and an image of the current observation. "\
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})

        self.reset()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]


    def predict_inference(self):

        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
        )
        output_text = outputs.choices[0].message.content
        output_text = output_text.strip()
        
        return output_text

    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()

        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None
    

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self,action_id):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")
        
    def save_episode_video(self):
        if not self.save_gif or len(self.topdown_map_list) == 0:
            return
        output_video_path = os.path.join(
            self.result_path, "video", "{}.gif".format(self.episode_id))
        imageio.mimsave(output_video_path, self.topdown_map_list)

    def reset(self):
        self.topdown_map_list = []

        self.pending_action_list = []
        self.rgb_list = []

        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        
    def act(self, observations, info, episode_id, env=None):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        rgb_ = rgb_.resize((308,252))
        self.rgb_list.append(rgb_)
        # do not cut down rgb list while using uniform sampling
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            return {"action": temp_action}

        # for observation1+observation2 action style
        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            content.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(item)}"}} for item in self.uniform_sample_with_ends(self.rgb_list[:-1],8)])
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation')
        instruction_suffix = item[1]
        if self.use_goal_hint:
            if env is None:
                raise ValueError("env is required when --use-goal-hint is enabled")
            current_pose = agent_state_to_pose(env.sim.get_agent_state())
            goal_pose = {"position": list(env.current_episode.goals[0].position)}
            instruction_suffix = f"{instruction_suffix.strip()} {format_goal_hint(current_pose, goal_pose)}"
        content.append({"type": "text", "text": instruction_suffix})


        self.conversations.append({
                "role": "user",
                "content": content
            })

        navigation = self.predict_inference()
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)
        
        result = self.extract_multi_result(navigation)

        select_action_idx = 2

        result = result[:select_action_idx]
        for action_index,numeric in result:

            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, round(numeric/self.forward_distance))):
                    self.pending_action_list.append(1)

            elif action_index == 2:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(2)

            elif action_index == 3:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(3)
            
            if action_index is None or len(self.pending_action_list)==0:
                print('random select an action')
                action_index = random.randint(1, 3)
                navigation = self.action_id_to_str(action_index)
                self.pending_action_list.append(action_index)

        return {"action": self.pending_action_list.pop(0)}


def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--split-id",type=int,default=None,help="chunk ID; if set, run only this shard (single process)")
    parser.add_argument("--resolution-ratio",type=float,help="location of model weights",default=0.5)
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument("--max-action-history",type=int,help="the maximum num of action history",default=10)
    parser.add_argument("--num-generations",type=int,help="whether use video or multi image",default=1)
    parser.add_argument("--max-episodes",type=int,default=None,help="maximum number of episodes to evaluate")
    parser.add_argument(
        "--episode-ids-file",
        type=str,
        default=None,
        help="path to a file with one episode id per line; only evaluate these episodes (re-evaluates even if results exist)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        help="dataset split to evaluate, e.g. val_unseen, val_seen, train",
    )
    parser.add_argument(
        "--use-goal-hint",
        type=str2bool,
        default=False,
        help="append oracle goal position hint to the instruction prompt (matches goal training data)",
    )
    parser.add_argument(
        "--save-gif",
        type=str2bool,
        default=True,
        help="whether to save episode navigation visualization as .gif under result-path/video/",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None

    config = get_config(args.exp_config)
    with habitat.config.read_write(config):
        if args.dataset_split is not None:
            config.habitat.dataset.split = args.dataset_split
        # self.config.habitat.task.measurements.success.success_distance=3.0
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
            
    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)

    use_episode_subset = args.episode_ids_file is not None
    if use_episode_subset:
        filter_dataset_by_episode_ids(dataset, args.episode_ids_file)
    if args.max_episodes is not None:
        dataset.episodes = dataset.episodes[:args.max_episodes]
    if len(dataset.episodes) == 0:
        raise ValueError("No episodes to evaluate.")

    split_num = min(args.split_num, len(dataset.episodes))
    dataset_splits = dataset.get_splits(split_num, allow_uneven_splits=True)

    worker_args_base = (
        args.num_generations, args.forward_distance, args.turn_angle,
        args.max_action_history, args.resolution_ratio,
        args.use_goal_hint, args.save_gif, use_episode_subset,
    )

    if args.split_id is not None:
        if args.split_id < 0 or args.split_id >= split_num:
            raise ValueError(f"split-id must be in [0, {split_num}), got {args.split_id}")
        result_queue = queue.Queue()
        evaluate_agent(
            result_queue, api_key, base_url, config, dataset_splits[args.split_id],
            args.result_path, *worker_args_base,
        )
        return

    num_episodes = len(dataset.episodes)

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(split_num):
        worker_args = (result_queue, api_key, base_url, config, dataset_splits[i], args.result_path,
                *worker_args_base)
        p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()