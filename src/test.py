import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Semaphore pour limiter le nombre de tâches simultanées
sem = threading.Semaphore(5)

def gpu_task(x):
    print(f"Processing {x}")
    time.sleep(2)  # simule un calcul sur GPU
    return x * x

def callback(future):
    # sem.release()
    print("callback")
    result = future.result()  # récupère le résultat
    print(f"Task finished with result: {result}")

# Thread pool
with ThreadPoolExecutor(max_workers=3) as executor:
    for i in range(10):
        # sem.acquire()
        print(i)
        future = executor.submit(gpu_task, i)
        future.add_done_callback(callback)  # exécute le callback dès que terminé

print("All tasks submitted, main thread is free to do other work")
time.sleep(5)  # juste pour garder le script ouvert pour voir les callbacks
