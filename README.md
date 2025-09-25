# Monitor: AI-Powered Network Intrusion Detection System (NIDS) 🔒🛡️

A prototype security system that uses unsupervised machine learning to identify malicious network activity like DDoS attacks and port scanning. This end-to-end project demonstrates the application of AI to cybersecurity, from feature engineering to an interactive analyst dashboard.

> **Key Achievement:** Built an anomaly detection pipeline that achieves **63% recall** in identifying attacks while maintaining **61% precision**, providing a viable baseline for real-world security monitoring.

## 🚀 Features

- **Automated Threat Detection:** Flags suspicious network flows in real-time using an Isolation Forest model trained on benign traffic.
- **Interactive Security Dashboard:** A Streamlit web application that allows security analysts to upload data, tune sensitivity, and review prioritized alerts.
- **End-to-End ML Pipeline:** Handles everything from data preprocessing and feature engineering to model training, evaluation, and deployment.

## 📸 Demonstration

![Monitor NIDS Dashboard](assets/dashboard-screenshot282.png)
*The Streamlit dashboard allows analysts to interact with the detection system, adjusting the anomaly threshold and inspecting flagged flows.*

**Example Detection:**
- **Input:** Network flow data containing a port scan.
- **Output:** The system flags the suspicious IP address, highlighting the anomalous packet patterns for analyst review.

## 📊 Performance & Results

The system was designed for the practical reality of cybersecurity: attacks are rare and diverse. The chosen configuration optimizes for catching threats while managing false alarms.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Recall** | 0.63 | Detects **63% of all attacks** in the test set. |
| **Precision** | 0.61 | **61% of its alerts are true threats**, minimizing analyst time wasted on false alarms. |
| **F1-Score** | 0.62 | A balanced measure of the model's overall effectiveness. |

**Model Choice Justification:** Selected Isolation Forest for its superiority in anomaly detection tasks—it efficiently isolates outliers without needing labeled attack data, mirroring real-world security operations.

## 🛠️ Tech Stack

- **Machine Learning:** Scikit-learn (Isolation Forest, StandardScaler)
- **Backend & Processing:** Python, Pandas, NumPy
- **Web Dashboard:** Streamlit
- **MLOps:** Joblib (model serialization)

## 🧭 Getting Started

### Prerequisites
- Python 3.8+

### Installation & Run

1.  **Clone and set up the environment:**
    ```bash
    git clone https://github.com/Auction-s/Monitor.git
    cd Monitor
    python -m venv anomaly_env
    source anomaly_env/bin/activate  # Linux/macOS: `source anomaly_env/bin/activate` | Windows: `.\anomaly_env\Scripts\activate`
    pip install -r requirements.txt
    ```

2.  **Train the model on your network flow data:**
    ```bash
    python src/train_models.py --input data/flows_features.csv --out_dir ./models --contamination 0.25
    ```

3.  **Launch the interactive dashboard:**
    ```bash
    streamlit run src/stream.py
    ```

## 🔬 Implementation Highlights

- **Feature Engineering:** Created domain-specific features like `bytes/sec`, `packets/sec`, and `forward-to-backward packet ratios` that are critical for identifying anomalous behavior.
- **Unsupervised Approach:** The model is trained only on "benign" traffic, making it applicable in real-world scenarios where labeled attack data is scarce.
- **Production-Ready Code:** Structured into modular scripts for preprocessing, training, and inference, ensuring reproducibility and ease of integration.

## 🚀 Future Improvements

- Develop ensemble methods combining Isolation Forest with other algorithms (e.g., Autoencoders) to improve detection rates.
- Deploy the model as a REST API for integration with existing Security Information and Event Management (SIEM) systems.
- Implement real-time detection capabilities using streaming data platforms like Apache Kafka.

---

## 📬 Contact

**Ayanfeoluwa Olateju**
- GitHub: [Auction-s](https://github.com/Auction-s)
- Email: [olatejuaayanfeoluwa@gmail.com]
- Portfolio: [https://ayanfe-portfolio.vercel.app/]  
