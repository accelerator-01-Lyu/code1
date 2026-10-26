import os
HERE = r"D:\AL_FuzzyClustering_Paper\v2_unified\code\results"
done = open(os.path.join(HERE, "shards", "done.txt")).read().splitlines()
p3 = [l for l in done if "/P3/" in l]
p4 = [l for l in done if "/P4/" in l]
flag = os.path.join(HERE, ".main_worker1.flag")
print(f"P3 shards = {len(p3)} / 1440")
print(f"P4 shards = {len(p4)} / 720")
print(f".main_worker1.flag exists = {os.path.exists(flag)}")
ready = (len(p3) >= 1440 and len(p4) >= 720 and os.path.exists(flag))
print("READY" if ready else "WAIT")
