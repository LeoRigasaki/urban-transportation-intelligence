"""
Live Streaming Monitor for Lahore Traffic Intelligence.
Watch real-time traffic predictions flowing through the Kafka pipeline.
"""
import json
import time
import logging
from kafka import KafkaConsumer
from datetime import datetime

# Configuration
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "lahore_traffic_predictions"

def run_monitor():
    print("=" * 70)
    print("🚦 LAHORE TRAFFIC INTELLIGENCE: REAL-TIME PREDICTION MONITOR")
    print("=" * 70)
    print(f"Listening on topic: {TOPIC}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    print(f"{'TIMESTAMP':<20} | {'EDGE (U-V)':<20} | {'SPEED':<7} | {'PRED':<7} | {'LATENCY':<8}")
    print("-" * 70)

    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='live_monitor_group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        for message in consumer:
            data = message.value
            ts = datetime.fromtimestamp(data.get('prediction_timestamp', time.time())).strftime('%H:%M:%S')
            edge = f"{data.get('u', 0)}->{data.get('v', 0)}"
            curr_speed = data.get('current_speed', 0.0)
            pred_speed = data.get('predicted_speed_15min', 0.0)
            latency = f"{data.get('latency_ms', 0.0):.2f}ms"

            print(f"{ts:<20} | {edge:<20} | {curr_speed:<7.2f} | {pred_speed:<7.2f} | {latency:<8}")

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("🛑 Monitor stopped by user.")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    run_monitor()
