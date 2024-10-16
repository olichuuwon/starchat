import streamlit as st
import requests
import time


# Function to fetch and parse the Triton Server metrics from the /metrics endpoint
def fetch_and_parse_metrics():
    try:
        # Fetch metrics from the Triton server
        response = requests.get(
            "http://monitoring-triton-inference-services.apps.nebula.sl/metrics"
        )
        metrics_raw = response.text

        # Parse the metrics (in Prometheus format) into a dictionary
        metrics = {}
        for line in metrics_raw.splitlines():
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            key_value = line.split(" ")
            if len(key_value) == 2:
                key, value = key_value
                metrics[key] = float(value)

        return metrics
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return {}


import streamlit as st
import time


# Function to display all the metrics in the dashboard using placeholders
def display_metrics_with_placeholders(placeholders, metrics):
    # Display successful inferences
    placeholders["success"].metric(
        label="Successful Inferences",
        value=metrics.get(
            'nv_inference_request_success{model="meta-llama-3-8b-instruct-awq",version="1"}',
            "N/A",
        ),
    )

    # Display failed inferences
    placeholders["fail_other"].metric(
        label="Failed Inferences (Other)",
        value=metrics.get(
            'nv_inference_request_failure{model="facebook-opt125m",reason="OTHER",version="1"}',
            0,
        ),
    )
    placeholders["fail_backend"].metric(
        label="Failed Inferences (Backend)",
        value=metrics.get(
            'nv_inference_request_failure{model="meta-llama-3-8b-instruct-awq",reason="BACKEND",version="1"}',
            0,
        ),
    )

    # GPU Utilization
    gpu_util_key = (
        'nv_gpu_utilization{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
    )
    placeholders["gpu_util"].metric(
        label="GPU Utilization", value=f"{metrics.get(gpu_util_key, 0) * 100:.2f}%"
    )

    # Inference request duration
    duration_key = 'nv_inference_request_duration_us{model="meta-llama-3-8b-instruct-awq",version="1"}'
    placeholders["request_duration"].metric(
        label="Inference Request Duration (µs)",
        value=f"{metrics.get(duration_key, 0):,.0f}",
    )

    # GPU memory usage
    gpu_memory_used_key = (
        'nv_gpu_memory_used_bytes{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
    )
    gpu_memory_total_key = (
        'nv_gpu_memory_total_bytes{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
    )
    used_memory = metrics.get(gpu_memory_used_key, 0)
    total_memory = metrics.get(gpu_memory_total_key, 1)
    memory_percentage = (used_memory / total_memory) * 100
    placeholders["gpu_memory"].metric(
        label="GPU Memory Usage",
        value=f"{used_memory / 1e9:.2f} GB",
        delta=f"{memory_percentage:.2f}%",
    )

    # CPU Utilization
    placeholders["cpu_util"].metric(
        label="CPU Utilization",
        value=f"{metrics.get('nv_cpu_utilization', 0) * 100:.2f}%",
    )

    # GPU Power Usage
    power_usage_key = (
        'nv_gpu_power_usage{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
    )
    power_limit_key = (
        'nv_gpu_power_limit{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
    )
    power_usage = metrics.get(power_usage_key, 0)
    power_limit = metrics.get(power_limit_key, "N/A")
    placeholders["gpu_power"].metric(
        label="GPU Power Usage (Watts)",
        value=f"{power_usage} W",
        delta=f"Limit: {power_limit} W",
    )

    # Energy Consumption
    energy_key = (
        'nv_energy_consumption{gpu_uuid="GPU-47385698-1baf-7602-e480-a211177a1640"}'
    )
    placeholders["energy"].metric(
        label="Energy Consumption (Joules)",
        value=f"{metrics.get(energy_key, 0):,.0f} J",
    )

    # CPU Memory Usage
    cpu_memory_used_key = "nv_cpu_memory_used_bytes"
    cpu_memory_total_key = "nv_cpu_memory_total_bytes"
    used_cpu_memory = metrics.get(cpu_memory_used_key, 0)
    total_cpu_memory = metrics.get(cpu_memory_total_key, 1)
    cpu_memory_percentage = (used_cpu_memory / total_cpu_memory) * 100
    placeholders["cpu_memory"].metric(
        label="CPU Memory Usage",
        value=f"{used_cpu_memory / 1e9:.2f} GB",
        delta=f"{cpu_memory_percentage:.2f}%",
    )

    # Display pending request count for the model
    pending_requests_key = 'nv_inference_pending_request_count{model="meta-llama-3-8b-instruct-awq",version="1"}'
    placeholders["pending_requests"].metric(
        label="Pending Requests", value=metrics.get(pending_requests_key, 0)
    )


# Initialize placeholders for each metric, adding one for pending requests
placeholders = {
    "success": st.empty(),
    "fail_other": st.empty(),
    "fail_backend": st.empty(),
    "gpu_util": st.empty(),
    "request_duration": st.empty(),
    "gpu_memory": st.empty(),
    "cpu_util": st.empty(),
    "gpu_power": st.empty(),
    "energy": st.empty(),
    "cpu_memory": st.empty(),
    "pending_requests": st.empty(),  # New placeholder for pending requests
}

# Real-time monitoring loop
if st.button("Start Monitoring"):
    st.write("Fetching metrics every 5 seconds...")

    with st.spinner("Fetching metrics..."):
        while True:  # Continuous monitoring
            metrics = fetch_and_parse_metrics()
            display_metrics_with_placeholders(placeholders, metrics)
            time.sleep(5)  # Refresh every 5 seconds
else:
    st.write("Press the button to start monitoring.")
