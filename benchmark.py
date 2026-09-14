"""Optional local CPU benchmark; uses no camera or experimental measurements."""
import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from core import MODEL


if __name__ == '__main__':
    data = np.zeros((1, 3, 960, 960), np.float32)
    for threads in [1, 2, 4, 8]:
        interpreter = Interpreter(model_path=str(MODEL), num_threads=threads)
        interpreter.allocate_tensors()
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], data)
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            interpreter.invoke()
            durations.append(time.perf_counter()-start)
        print(f'Threads={threads}: median invocation {np.median(durations):.3f}s')
