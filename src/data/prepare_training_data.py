import json
import os
import random
import re
import copy
from tqdm import tqdm
import argparse
import numpy as np
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector


forward_distance = 25
turn_angle = 15

def noise_action(action_id):
    p = random.random()
    if p < 0.7 and action_id != 0:
        return random.randint(1,3)
    else:
        return action_id
def action_id_to_str(action_id):
    # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
    if action_id == 0:
        return "stop"
    elif action_id == 1:
        return f"forward {forward_distance} cm"
    elif action_id == 2:
        return f"turn left {turn_angle} degree"
    elif action_id == 3:
        return f"turn right {turn_angle} degree"
    else:
        raise ValueError(f"Invalid action ID: {action_id}")

def combine(action1,action2):
    idx = action1.rfind(', ')
    subaction0 = action1[:idx+1]+' ' if action1[:idx+1] != '' else action1[:idx+1]
    subaction1 = action1[idx+1:]
    match1 = re.search(r'-?\d+', subaction1)
    match1 = int(match1.group())
    match2 = re.search(r'-?\d+', action2)
    match2 = int(match2.group())
    if "forward" in subaction1:
        if match1+match2 <= 3*forward_distance:
            return f"{subaction0}forward {match1+match2} cm"
        else:
            return None
    elif "turn left" in subaction1:
        if match1+match2 <= 3*turn_angle:
            return f"{subaction0}turn left {match1+match2} degree"
        else:
            return None
    elif "turn right" in subaction1:
        if match1+match2 <= 3*turn_angle:
            return f"{subaction0}turn right {match1+match2} degree"
        else:
            return None
    else:
        raise ValueError(f"Invalid action: {action1}")



def spilt_method(x):
    return x.split("_")[1].split(".")[0]

def spilt_method_2(x):
    return x.split(".")[0]


def frame_idx_from_image(image_path, split_method):
    return int(split_method(os.path.basename(image_path)))


config = {
    'r2r': {
        'image_path': 'data/images/r2r',
        'annotation_path': 'data/sub_dataset/r2r.jsonl',
        'pose_path': 'data/poses/r2r',
        'split_method': spilt_method,
    },
    'rxr': {
        'image_path': 'data/images/rxr',
        'annotation_path': 'data/sub_dataset/rxr.jsonl',
        'pose_path': 'data/poses/rxr',
        'split_method': spilt_method,
    },
    'streamvln_rxr': {
        'image_path': 'data/streamvln/RxR',
        'annotation_path': 'data/sub_dataset/streamvln_rxr.jsonl',
        'split_method': spilt_method_2,
        'sub_image_path':' rgb',
    },
    'envdrop': {
        'image_path': 'data/images/envdrop',
        'annotation_path': 'data/sub_dataset/envdrop.jsonl',
        'pose_path': 'data/poses/envdrop',
        'split_method': spilt_method,
    },
    'scalevln': {
        'image_path': 'data/images/scalevln',
        'annotation_path': 'data/sub_dataset/scalevln.jsonl',
        'pose_path': 'data/poses/scalevln',
        'split_method': spilt_method,
    },
}


def angle_to_direction_word(angle_deg):
    angle_deg = (angle_deg + 360) % 360
    if angle_deg <= 22.5 or angle_deg > 337.5:
        return "ahead"
    if angle_deg <= 67.5:
        return "front-left"
    if angle_deg <= 112.5:
        return "left"
    if angle_deg <= 157.5:
        return "back-left"
    if angle_deg <= 202.5:
        return "behind"
    if angle_deg <= 247.5:
        return "back-right"
    if angle_deg <= 292.5:
        return "right"
    return "front-right"


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
    distance = float(np.linalg.norm(direction_vector[[0, 2]]))
    direction_word = angle_to_direction_word(
        float(np.degrees(np.arctan2(y_lateral, x_forward)))
    )

    return (
        f"Goal hint: Relative to the robot, the goal is {x_abs:.1f} m {front_or_back} "
        f"and {y_abs:.1f} m to the {left_or_right}. Distance: {distance:.1f} m. "
        f"Direction: {direction_word}."
    )


def load_episode_poses(pose_path, episode_id):
    episode_pose_path = os.path.join(pose_path, f"{episode_id}.json")
    with open(episode_pose_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)
    return pose_data["poses"]


def process_single_type(selected_subset_list, system_prompt, prompt_template, task_type):
    data2save = []
    for subset in tqdm(selected_subset_list):

        subset_config = config[subset]
        image_path = subset_config['image_path']
        annotation_path = subset_config['annotation_path']
        split_method = subset_config['split_method']
        sub_image_path = subset_config.get('sub_image_path', None)
        pose_path = subset_config.get('pose_path', None)
        if task_type == "goal" and pose_path is None:
            raise ValueError(f"Dataset {subset} does not have pose_path configured for goal task type.")

        # read annotations
        annotation = []
        with open(annotation_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                annotation.append(item)
            
        for episode_item in annotation:
            episode_id = episode_item['episode_id']
            video_id = episode_item['video_id']
            actions = episode_item['actions']
            assert actions[-1] == 0

            episode_poses = None
            goal_pose = None
            if task_type == "goal":
                episode_poses = load_episode_poses(pose_path, episode_id)
                goal_pose = episode_poses[-1]

            if isinstance(episode_item['instruction'], list):
                instruction_list = episode_item['instruction']
            else:
                instruction_list = [episode_item['instruction']]
            if task_type == "idm":
                instruction_list = instruction_list[:1]
            
            for instruction in instruction_list:

                episode_image_path = os.path.join(image_path, str(video_id))
                if sub_image_path is not None:
                    episode_image_path = os.path.join(episode_image_path, sub_image_path)
                episode_image_list = os.listdir(episode_image_path)
                episode_image_list = sorted(episode_image_list, key=lambda x: int(split_method(x)))
                episode_image_list = [episode_image_path+'/'+image for image in episode_image_list]
                if len(episode_image_list) != len(actions) + 1:
                    episode_image_list.append(episode_image_list[-1])
                assert len(episode_image_list) == len(actions) + 1

                def build_user_instruction(step_idx):
                    formated_instruction = prompt_template.format(instruction)
                    if task_type == "goal":
                        current_pose = episode_poses[min(step_idx, len(episode_poses) - 1)]
                        goal_hint = format_goal_hint(current_pose, goal_pose)
                        formated_instruction = f"{formated_instruction} {goal_hint}"
                    return formated_instruction

                def build_user_instruction_from_images(image_list):
                    frame_idx = frame_idx_from_image(image_list[-1], split_method)
                    return build_user_instruction(frame_idx)

                tmp_data = {
                    "system": system_prompt,
                    "conversations": [],
                    "action_history":[],
                    "episode_id":str(episode_id),
                    # 'idm_input': [],
                    "task type": task_type,
                }
                if task_type in ("vln", "goal"):
                    tmp_data["conversations"].append({"from": "user", "value": build_user_instruction(0),"image":[episode_image_list[0]]})
                    tmp_data["conversations"].append({"from": "assistant", "value": action_id_to_str(actions[0])}) 
                    last_action = actions[0]
                    pending_rgb_list = []
                    for i in range(1, len(actions)):
                        prob = random.random()
                        if prob <= 0.7 and actions[i] == last_action and combine(tmp_data["conversations"][-1]['value'],action_id_to_str(last_action)) is not None:
                            tmp_data["conversations"][-1]['value'] = combine(tmp_data["conversations"][-1]['value'],action_id_to_str(last_action))
                            pending_rgb_list.append(episode_image_list[i])
                        else:
                            count = tmp_data["conversations"][-1]['value'].count(',')
                            if count < 2:
                                tmp_data["conversations"][-1]['value'] += ', ' + action_id_to_str(actions[i])
                                pending_rgb_list.append(episode_image_list[i])
                            else:
                                if task_type == "goal":
                                    tmp_data["conversations"][0]['value'] = build_user_instruction_from_images(
                                        tmp_data["conversations"][0]['image']
                                    )
                                # tmp_data['idm_input'] = [tmp_data["conversations"][0]['image'][-1], episode_image_list[i]]
                                data2save.append(copy.deepcopy(tmp_data))
                                tmp_data["action_history"].append(tmp_data["conversations"][1]['value'])
                                while len(pending_rgb_list)!= 0:
                                    item = pending_rgb_list.pop(0)
                                    tmp_data["conversations"][0]['image'].append(item)
                                tmp_data["conversations"][0]['image'].append(episode_image_list[i])
                                # while len(tmp_data["conversations"][0]['image']) > max_images:
                                #     tmp_data["conversations"][0]['image'] = tmp_data["conversations"][0]['image'][1:]
                                tmp_data["conversations"][1]['value'] = action_id_to_str(actions[i])
                                # assert len(conversation["action_history"]) == len(conversation["conversations"][0]['image']) - 1
                        last_action = actions[i]
                    if task_type == "goal":
                        tmp_data["conversations"][0]['value'] = build_user_instruction_from_images(
                            tmp_data["conversations"][0]['image']
                        )
                    # tmp_data['idm_input'] = [tmp_data["conversations"][0]['image'][-1], episode_image_list[i+1]]
                    data2save.append(copy.deepcopy(tmp_data))
                elif task_type == "idm":
                    formated_instruction = prompt_template
                    tmp_data["conversations"].append({"from": "user", "value": formated_instruction,"image":[episode_image_list[0], episode_image_list[1]]})
                    tmp_data["conversations"].append({"from": "assistant", "value": action_id_to_str(actions[0])}) 
                    last_action = actions[0]
                    for i in range(1, len(actions)):
                        prob = random.random()
                        if prob <= 0.7 and actions[i] == last_action and combine(tmp_data["conversations"][-1]['value'],action_id_to_str(last_action)) is not None:
                            tmp_data["conversations"][-1]['value'] = combine(tmp_data["conversations"][-1]['value'],action_id_to_str(last_action))
                            tmp_data["conversations"][-2]['image'][-1] = episode_image_list[i+1]
                        else:
                            count = tmp_data["conversations"][-1]['value'].count(',')
                            if count < 2:
                                tmp_data["conversations"][-1]['value'] += ', ' + action_id_to_str(actions[i])
                                tmp_data["conversations"][-2]['image'][-1] = episode_image_list[i+1]
                            else:
                                data2save.append(copy.deepcopy(tmp_data))
                                tmp_data["action_history"].append(tmp_data["conversations"][1]['value'])
                                tmp_data["conversations"][0]['image'] = [episode_image_list[i],episode_image_list[i+1]]
                                tmp_data["conversations"][1]['value'] = action_id_to_str(actions[i])
                                # assert len(conversation["action_history"]) == len(conversation["conversations"][0]['image']) - 1
                        last_action = actions[i]
                    if tmp_data["conversations"][1]['value'] != 'stop':
                        data2save.append(copy.deepcopy(tmp_data))
    return data2save

    

def main(selected_subset_list, task_type_list, output_path):
    system_prompt = "You are a helpful assistant."
    vln_prompt_template = "Imagine you are a robot programmed for navigation tasks. "\
        "You have been given a video of historical observations and an image of the current observation. "\
        "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
        "which could involve turning left or right by a specific degree or moving forward a certain distance."

    idm_prompt_template = "Imagine you are a robot programmed for navigation tasks. "\
        "You have been given an image of current view and an image of the goal view. "\
        "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, "\
        "which could involve turning left "\
        "or right by a specific degree or moving forward a certain distance."
    
    data2save = []

    for task_type in task_type_list:
        if task_type in ("vln", "goal"):
            prompt_template = vln_prompt_template
        elif task_type == "idm":
            prompt_template = idm_prompt_template
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        data2save.extend(process_single_type(selected_subset_list, system_prompt, prompt_template, task_type))

    print(f"total number of samples = {len(data2save)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data2save:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(len(data2save))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r",],
    )
    parser.add_argument(
        "--task_type",
        nargs="+",
        default=["vln", "idm"],
    )
    parser.add_argument("--output_path", type=str, default='data/navida_train_data.jsonl')
    args = parser.parse_args()


    main(args.dataset_name, args.task_type, args.output_path)