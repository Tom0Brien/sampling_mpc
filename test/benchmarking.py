#!/usr/bin/env python3
import time
import numpy as np
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from jax import profiler

def setup_model_data():
    xml_path = "models/franka_emika_panda/mjx_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return mjx.put_model(model), mjx.put_data(model, data)

def benchmark_batch_step(model, data, batch_size=1024, n_steps=100):
    """Benchmark batched execution using vmap."""
    # Create batched data
    batch = jax.tree.map(
        lambda x: jp.repeat(x[None], batch_size, axis=0), 
        data
    )
    
    # Compile batched step and measure compile time
    vstep = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    
    # Measure compile time (first call includes compilation)
    start_compile = time.perf_counter()
    vstep(model, batch).qpos.block_until_ready()
    compile_time = time.perf_counter() - start_compile
    
    # Warmup (optional, to ensure any initialization is complete)
    vstep(model, batch).qpos.block_until_ready()
    
    # Time batch steps
    times = []
    for _ in range(5):  # Multiple measurements
        start = time.perf_counter()
        batch = vstep(model, batch)
        batch.qpos.block_until_ready()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    print("\nBatched Step Results:")
    print(f"JIT Compile time: {compile_time*1000:.3f}ms")
    print(f"Batch size: {batch_size}")
    print(f"Average step time: {avg_time*1000:.3f}ms")
    print(f"Total steps/sec: {(batch_size*n_steps)/avg_time:.0f}")
    print(f"Per instance: {avg_time/batch_size*1000:.3f}ms/instance")

def benchmark_batch_forward(model, data, batch_size=1024, n_steps=100):
    """Benchmark batched execution using vmap."""
    # Create batched data
    batch = jax.tree.map(
        lambda x: jp.repeat(x[None], batch_size, axis=0), 
        data
    )
    
    # Compile batched forward and measure compile time
    vforward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
    
    # Measure compile time (first call includes compilation)
    start_compile = time.perf_counter()
    vforward(model, batch).qpos.block_until_ready()
    compile_time = time.perf_counter() - start_compile
    
    # Warmup (optional, to ensure any initialization is complete)
    vforward(model, batch).qpos.block_until_ready()
    
    # Time batch forward passes
    times = []
    for _ in range(5):  # Multiple measurements
        start = time.perf_counter()
        batch = vforward(model, batch)
        batch.qpos.block_until_ready()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    print("\nBatched Forward Results:")
    print(f"JIT Compile time: {compile_time*1000:.3f}ms")
    print(f"Batch size: {batch_size}")
    print(f"Average forward time: {avg_time*1000:.3f}ms")
    print(f"Total forward/sec: {(batch_size*n_steps)/avg_time:.0f}")
    print(f"Per instance: {avg_time/batch_size*1000:.3f}ms/instance")

if __name__ == "__main__":
    # Initial setup
    model, data = setup_model_data()
    
    # Run benchmarks
    benchmark_batch_step(model, data)
    benchmark_batch_forward(model, data)