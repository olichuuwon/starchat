import streamlit as st
import time
import requests


# Function to fetch and parse metrics
def fetch_and_parse_metrics():
    # Fetch metrics from the Triton server
    url = "http://monitoring-triton-inference-services.apps.nebula.sl/metrics"  # Adjust the URL if needed
    response = requests.get(url)
    metrics_raw = response.text
    metrics = {}
    for line in metrics_raw.splitlines():
        if not line.startswith("#"):
            key, value = line.split()
            metrics[key] = float(value)
    return metrics


# Function to display metrics in a 4x3 grid layout
def display_metrics(metrics):
    # Create a 4x3 grid layout using Streamlit columns
    cols = st.columns(3)

    # Row 1
    with cols[0]:
        st.metric(
            label="GPU Utilization",
            value=f"{metrics.get('nv_gpu_utilization{gpu_uuid=\"GPU-47385698-1baf-7602-e480-a211177a1640\"}', 0) * 100:.2f}%",
        )
    with cols[1]:
        st.metric(
            label="CPU Utilization",
            value=f"{metrics.get('nv_cpu_utilization', 0) * 100:.2f}%",
        )
    with cols[2]:
        pending_requests_key = 'nv_inference_pending_request_count{model="meta-llama-3-8b-instruct-awq",version="1"}'
        st.metric(label="Pending Requests", value=metrics.get(pending_requests_key, 0))

    # Row 2
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            label="Successful Inferences",
            value=metrics.get(
                'nv_inference_request_success{model="meta-llama-3-8b-instruct-awq",version="1"}',
                "N/A",
            ),
        )
    with cols[1]:
        st.metric(
            label="Failed Inferences (Other)",
            value=metrics.get(
                'nv_inference_request_failure{model="facebook-opt125m",reason="OTHER",version="1"}',
                0,
            ),
        )
    with cols[2]:
        st.metric(
            label="Failed Inferences (Backend)",
            value=metrics.get(
                'nv_inference_request_failure{model="meta-llama-3-8b-instruct-awq",reason="BACKEND",version="1"}',
                0,
            ),
        )

    # Row 3
    cols = st.columns(3)
    with cols[0]:
        gpu_memory_used_key = 'nv_gpu_memory_used_bytes{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
        gpu_memory_total_key = 'nv_gpu_memory_total_bytes{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
        used_memory = metrics.get(gpu_memory_used_key, 0)
        total_memory = metrics.get(gpu_memory_total_key, 1)
        st.metric(
            label="GPU Memory Usage",
            value=f"{used_memory / 1e9:.2f} GB",
            delta=f"{(used_memory / total_memory) * 100:.2f}%",
        )
    with cols[1]:
        cpu_memory_used_key = "nv_cpu_memory_used_bytes"
        cpu_memory_total_key = "nv_cpu_memory_total_bytes"
        used_cpu_memory = metrics.get(cpu_memory_used_key, 0)
        total_cpu_memory = metrics.get(cpu_memory_total_key, 1)
        st.metric(
            label="CPU Memory Usage",
            value=f"{used_cpu_memory / 1e9:.2f} GB",
            delta=f"{(used_cpu_memory / total_cpu_memory) * 100:.2f}%",
        )
    with cols[2]:
        power_usage_key = (
            'nv_gpu_power_usage{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
        )
        power_limit_key = (
            'nv_gpu_power_limit{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
        )
        power_usage = metrics.get(power_usage_key, 0)
        power_limit = metrics.get(power_limit_key, "N/A")
        st.metric(
            label="GPU Power Usage (Watts)",
            value=f"{power_usage} W",
            delta=f"Limit: {power_limit} W",
        )

    # Row 4
    cols = st.columns(3)
    with cols[0]:
        duration_key = 'nv_inference_request_duration_us{model="meta-llama-3-8b-instruct-awq",version="1"}'
        st.metric(
            label="Inference Request Duration (µs)",
            value=f"{metrics.get(duration_key, 0):,.0f}",
        )
    with cols[1]:
        energy_key = (
            'nv_energy_consumption{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
        )
        st.metric(
            label="Energy Consumption (Joules)",
            value=f"{metrics.get(energy_key, 0):,.0f} J",
        )
    with cols[2]:
        st.metric(
            label="Execution Count",
            value=metrics.get(
                'nv_inference_exec_count{model="meta-llama-3-8b-instruct-awq",version="1"}',
                0,
            ),
        )


# Initialize the state to manage the monitoring
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

# Start/stop monitoring button
if st.button("Start Monitoring"):
    st.session_state.monitoring = True

# Display metrics and rerun the app if monitoring is active
if st.session_state.monitoring:
    st.write("Fetching metrics every 3 seconds...")
    metrics = fetch_and_parse_metrics()
    display_metrics(metrics)

    # Schedule a rerun every 3 seconds
    time.sleep(3)
    st.rerun()
else:
    st.write("Press the button to start monitoring.")
