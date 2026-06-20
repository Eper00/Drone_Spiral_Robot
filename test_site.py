import time
import numpy as np
from common.base_env import BaseEnv   # <-- ha más a fájlnév, írd át
from common.support import load_config

# -------------------------
# CONFIG PATH
# -------------------------
CONFIG_PATH = "/home/tomi/SpiralRobot/configs/default_rl_training.yaml"   # <-- írd át a sajátodra


def main():

    print("🔧 BaseEnv teszt indul...")

    # 1) ENV létrehozása
    env = BaseEnv(config=CONFIG_PATH, render_mode="human")
    print("✔ BaseEnv példányosítva")

    # 2) RESET
    env._base_reset()
    print("✔ Reset lefutott")

    # 3) OBS lekérése
    obs = env._get_current_raw_obs()
    print("✔ Observation lekérve")

    print(f"Observation shape: {obs.shape}")
    print(f"Első 10 elem: {obs[:10]}")

    # 4) Marker pozíciók ellenőrzése
    print("\n📍 Marker pozíciók (y,z):")
    print(env.marker_positions_tentacle,env.marker_positions_drone)

    # 5) Drón pozíció ellenőrzése
    print("\n🚁 Drón pozíció:")
    print(env.drone_pos)

    # 6) Goal pozíció ellenőrzése
    print("\n🎯 Goal pozíció:")
    print(env.goal_position)

    # 7) STEP teszt
    print("\n➡ Step teszt...")
    action = np.ones(env.actuator_dim)
    ok = env._base_step(action)

    if ok:
        print("✔ Step sikeres")
    else:
        print("❌ Step instabil volt")

    # 8) Viewer futtatása 3 másodpercig
    print("\n👀 Viewer fut 3 másodpercig...")
    start = time.time()
    while time.time() - start < 0.2:
        env.render()
        env._base_step(action)
    
    env.close()
    print("✔ Teszt kész")


if __name__ == "__main__":
    main()
