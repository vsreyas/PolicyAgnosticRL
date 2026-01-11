from jaxrl_m.envs.libero import time_limit, STEP_TIME_LIMIT, StepTimeout

try:
    with time_limit(60):
        while True:
            pass
except: # StepTimeout:
    print("Timed out")
    breakpoint()
