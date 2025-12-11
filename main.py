import carla
import math
import csv
import argparse
import os
import random
import sys
import subprocess
import time
from typing import List, Optional


N_NEIGHBORS = 50

def get_speed(vehicle: carla.Actor) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def world_to_ego(ego_tf: carla.Transform, loc: carla.Location):
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y

    yaw = math.radians(ego_tf.rotation.yaw)
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)

    x_ego = dx * cos_y - dy * sin_y
    y_ego = dx * sin_y + dy * cos_y
    return x_ego, y_ego


def choose_random_weather() -> carla.WeatherParameters:
    presets = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.SoftRainSunset,
        carla.WeatherParameters.HardRainSunset,
    ]
    return random.choice(presets)


def safe_destroy(actor: Optional[carla.Actor]):
    if actor is None:
        return
    try:
        if hasattr(actor, "is_alive"):
            if actor.is_alive:
                actor.destroy()
        else:
            actor.destroy()
    except Exception:
        pass


def spawn_ego(world, bp_lib, spawn_points, tm, max_trials=30) -> Optional[carla.Actor]:
    ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    random.shuffle(spawn_points)

    for _ in range(max_trials):
        sp = random.choice(spawn_points)
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego is not None:
            ego.set_autopilot(True, tm.get_port())
            return ego
    return None


def spawn_traffic(
    world,
    bp_lib,
    spawn_points,
    ego,
    tm,
    min_veh,
    max_veh,
    seed,
) -> List[carla.Actor]:
    random.seed(seed)
    vehicles: List[carla.Actor] = []

    ego_loc = ego.get_transform().location
    vehicle_bps = bp_lib.filter("vehicle.*")
    random.shuffle(spawn_points)

    num_target = random.randint(min_veh, max_veh)
    num_target = min(num_target, max(0, len(spawn_points) - 5))

    for sp in spawn_points:
        if len(vehicles) >= num_target:
            break

        if sp.location.distance(ego_loc) < 8.0:
            continue

        bp = random.choice(vehicle_bps)
        try:
            v = world.try_spawn_actor(bp, sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                vehicles.append(v)
        except Exception:
            continue

    print(f"  Spawned {len(vehicles)} traffic vehicles.")
    return vehicles

def run_single_episode(args):
    print("=" * 60)
    print(f"[Town {args.town}] Episode {args.episode_idx}")
    print("=" * 60)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    # load_world 재시도 로직
    world = None
    for trial in range(3):
        try:
            world = client.load_world(args.town)
            break
        except RuntimeError as e:
            print(f"  load_world timeout (trial {trial+1}/3): {e}")
            if trial == 2:
                print("  Failed to load world after retries, skip episode.")
                return
            time.sleep(5.0)

    if world is None:
        print("  World is None after retries, skip episode.")
        return

    bp_lib = world.get_blueprint_library()
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    if not spawn_points:
        print("  No spawn points, exit.")
        return

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.delta_t
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed + args.episode_idx)

    ego = None
    camera = None
    traffic_vehicles: List[carla.Actor] = []
    csv_file = None

    town_dir_name = f"{args.out_prefix}_{args.town}"
    episode_dir = os.path.join(town_dir_name, f"episode_{args.episode_idx:03d}")
    img_dir = os.path.join(episode_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(episode_dir, "trajectories.csv")

    try:
        weather = choose_random_weather()
        world.set_weather(weather)
        print("  Weather:", weather)

        ego = spawn_ego(world, bp_lib, spawn_points, tm)
        if ego is None:
            print("  Ego spawn failed, skip episode.")
            return
        print("  Ego id =", ego.id)

        # traffic
        traffic_vehicles = spawn_traffic(
            world,
            bp_lib,
            spawn_points,
            ego,
            tm,
            args.min_vehicles,
            args.max_vehicles,
            args.seed + args.episode_idx,
        )

        # 카메라
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.img_width))
        cam_bp.set_attribute("image_size_y", str(args.img_height))
        cam_bp.set_attribute("fov", "90")

        cam_tf = carla.Transform(
            carla.Location(x=1.5, z=2.2),
            carla.Rotation(pitch=0.0),
        )
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

        def save_img(img: carla.Image):
            img.save_to_disk(os.path.join(img_dir, f"{img.frame:06d}.png"))

        camera.listen(save_img)

        fieldnames = [
            "town",
            "episode",
            "frame",
            "time",
            "ego_x",
            "ego_y",
            "ego_yaw_deg",
            "ego_speed_mps",
            "image_file",
        ]
        for i in range(N_NEIGHBORS):
            fieldnames += [
                f"nbr{i}_x_ego",
                f"nbr{i}_y_ego",
                f"nbr{i}_speed_mps",
                f"nbr{i}_distance",
            ]
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames)
        writer.writeheader()

        frames = int(args.duration / args.delta_t)
        print(f"  Logging {frames} frames (~{args.duration}s @ {1/args.delta_t:.1f}Hz)")

        for step in range(frames):
            world.tick()
            snap = world.get_snapshot()
            ts = snap.timestamp

            if ego is None or (hasattr(ego, "is_alive") and not ego.is_alive):
                print("  Ego lost, end early.")
                break

            ego_tf = ego.get_transform()
            ego_speed = get_speed(ego)
            ego_yaw = ego_tf.rotation.yaw
            frame = snap.frame

            img_rel_path = os.path.join("images", f"{frame:06d}.png")

            row = {
                "town": args.town,
                "episode": args.episode_idx,
                "frame": frame,
                "time": ts.elapsed_seconds,
                "ego_x": ego_tf.location.x,
                "ego_y": ego_tf.location.y,
                "ego_yaw_deg": ego_yaw,
                "ego_speed_mps": ego_speed,
                "image_file": img_rel_path,
            }

            vehicles = world.get_actors().filter("vehicle.*")
            neighbors = []
            for v in vehicles:
                if v.id == ego.id:
                    continue
                try:
                    loc = v.get_transform().location
                except RuntimeError:
                    continue
                x_e, y_e = world_to_ego(ego_tf, loc)
                dist = math.hypot(x_e, y_e)
                speed = get_speed(v)
                neighbors.append((dist, x_e, y_e, speed))

            neighbors.sort(key=lambda x: x[0])

            for i in range(N_NEIGHBORS):
                if i < len(neighbors):
                    d, x_e, y_e, sp = neighbors[i]
                    row[f"nbr{i}_x_ego"] = x_e
                    row[f"nbr{i}_y_ego"] = y_e
                    row[f"nbr{i}_speed_mps"] = sp
                    row[f"nbr{i}_distance"] = d
                else:
                    row[f"nbr{i}_x_ego"] = ""
                    row[f"nbr{i}_y_ego"] = ""
                    row[f"nbr{i}_speed_mps"] = ""
                    row[f"nbr{i}_distance"] = ""

            writer.writerow(row)

            if step % 50 == 0:
                print(
                    f"    [{step}/{frames}] t={ts.elapsed_seconds:.1f}s, v={ego_speed:.2f}"
                )

        print(f"Saved: {csv_path}")

    finally:
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass
        print("  (single) Episode finished, skipping destroy for safety.")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm_port", type=int, default=8000)

    parser.add_argument("--mode", type=str, default="multi", choices=["multi", "single"])

    parser.add_argument(
        "--towns",
        nargs="*",
        default=["Town01", "Town02", "Town03", "Town04", "Town05"],
        help="multi 모드에서 사용할 맵 리스트",
    )
    parser.add_argument(
        "--episodes_per_town", type=int, default=20, help="맵당 에피소드 개수 (multi 모드)"
    )

    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--delta_t", type=float, default=0.1)
    parser.add_argument("--out_prefix", type=str, default="dataset")
    parser.add_argument("--min_vehicles", type=int, default=20)
    parser.add_argument("--max_vehicles", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_width", type=int, default=800)
    parser.add_argument("--img_height", type=int, default=600)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "single":
        random.seed(args.seed + args.episode_idx)
        run_single_episode(args)
        os._exit(0)

    print("Running in MULTI mode.")
    print("Towns:", args.towns)
    print("Episodes per town:", args.episodes_per_town)

    for town in args.towns:
        for ep in range(args.episodes_per_town):
            cmd = [
                sys.executable,
                sys.argv[0],
                "--mode", "single",
                "--host", args.host,
                "--port", str(args.port),
                "--tm_port", str(args.tm_port),
                "--town", town,
                "--episode_idx", str(ep),
                "--duration", str(args.duration),
                "--delta_t", str(args.delta_t),
                "--out_prefix", args.out_prefix,
                "--min_vehicles", str(args.min_vehicles),
                "--max_vehicles", str(args.max_vehicles),
                "--seed", str(args.seed),
                "--img_width", str(args.img_width),
                "--img_height", str(args.img_height),
            ]
            print("\n[Multi] Launching:", " ".join(cmd))
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"[Multi] Episode failed (town={town}, ep={ep}), returncode={ret.returncode}")
            else:
                print(f"[Multi] Episode finished (town={town}, ep={ep})")


if __name__ == "__main__":
    main()
