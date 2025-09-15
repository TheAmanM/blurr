# Blurr: Real-Time Privacy for the Creator Economy

![Image](https://github.com/user-attachments/assets/fe0e70b9-c3a6-45fc-b440-4a5b34fc555f)

_A preview of Blurr showing our software detecting and blurring sensitive information live._

## 🚀 Inspiration

In the creator economy, a single mistake can be catastrophic. Streamers, educators, and professionals live in constant fear of accidentally revealing a password, an API key, a phone number, or a private document during a live broadcast. A split-second error can lead to doxxing, financial loss, and a breach of trust with their audience. Existing solutions are manual and reactive—requiring streamers to use clumsy overlays or simply "be more careful." We knew there had to be a better way: a proactive, intelligent safety net that protects creators without them even thinking about it.

## 💡 What it does

**Blurr** is a cloud-based, AI-powered redaction service that acts as a real-time security guard for live video streams. By integrating directly with streaming platforms like Twitch, YouTube, and Discord, Blurr automatically detects and blurs sensitive information before it ever reaches the audience. For the streamer, it's as simple as flipping a switch in their broadcast settings.

* **Autonomous Redaction:** Intelligently identifies and blurs passwords, phone numbers, emails, API keys, private documents, and more.
* **Zero Performance Impact:** All AI processing happens on our powerful cloud infrastructure. This means **0% local CPU/GPU usage**, so creators can dedicate their PC's full power to gaming and encoding.
* **Seamless Integration:** Functions as a simple "Privacy Mode" toggle within the streaming platform's native dashboard. No downloads, no complicated setup.

### 🛠️ How we built it

Blurr is architected as a highly scalable, low-latency cloud service designed for real-time video processing.

* **Technology Stack:** The core is a **Python** backend leveraging state-of-the-art machine learning models like **YOLOv8** for object detection (documents, whiteboards) and optimized **OpenCV** pipelines with OCR for text-pattern recognition.
* **Low-Latency Transport:** We use **WebRTC** to transport video feeds between the streaming platform and our service. This peer-to-peer inspired protocol is critical for minimizing the latency introduced by our processing hop.
* **Cloud-Native Infrastructure:** The service runs on a **Kubernetes** cluster managing a fleet of GPU-enabled servers. We deployed this infrastructure across multiple global regions, creating **geo-distributed edge nodes**. This ensures that a streamer's video feed is always routed to the nearest data center, dramatically reducing round-trip time.
* **Server-Grade AI:** By running in the cloud, we can deploy much larger and more accurate AI models than would be feasible on a local machine. This allows us to detect a wider range of sensitive information with fewer false positives.

### ⚡ Challenges we ran into

1.  **The Sub-200ms Latency Barrier:** Our biggest challenge was making the round-trip to our cloud service imperceptible. This required meticulous optimization of our WebRTC implementation, strategic deployment of edge nodes, and streamlining our ML inference pipeline to process and return a 1080p video frame in milliseconds.
2.  **Earning Creator Trust:** Asking platforms to route raw video feeds through our third-party service required us to build an architecture founded on security. We designed our system to be ephemeral, processing streams entirely in-memory without ever writing data to disk.
3.  **The Accuracy Tightrope:** We had to fine-tune our models to be aggressive enough to catch fleeting glimpses of data, but not so aggressive that they would blur harmless game UI elements or text, which would ruin the viewing experience.

### 🏅 Accomplishments that we're proud of

* **Zero-Impact Security:** We successfully created a powerful security tool that has absolutely no performance cost for the end-user. This is a game-changer for streamers who cannot afford to sacrifice frames for safety.
* **Intelligent, Content-Aware Blurring:** Blurr is more than a simple filter. It demonstrates a contextual understanding of the screen, capable of redacting a single line of text in a document while leaving the rest perfectly readable.
* **Post-Stream "Guardian Report":** To provide tangible value and peace of mind, we developed a post-stream analytics report that shows creators exactly what information was protected and when, turning an abstract fear into a solved problem.

### 📚 What we learned

* **For Streamers, Performance is Everything:** Any solution that impacts stream quality is a non-starter. Offloading all heavy processing to the cloud is the only way to deliver a feature like this without compromise.
* **Edge Computing is the Future of Real-Time AI:** Centralized cloud processing is too slow for live interaction. A distributed, edge-based architecture is essential for building real-time AI services at a global scale.
* **Trust is Built Through Transparency:** In the security space, you can't just say "trust us." Providing users with clear, tangible proof of how you're protecting them is the most effective way to build confidence.

### 🚀 What's next for Blurr

* **Real-Time Audio Redaction:** Expanding our service to detect and bleep spoken sensitive information, like a credit card number read aloud.
* **Proactive Copyright Protection:** Building models to identify copyrighted music or video content in real-time, helping creators avoid DMCA strikes before they happen.
* **Enterprise Expansion:** Adapting Blurr for the corporate world to protect sensitive information during enterprise video calls and remote presentations on platforms like Zoom and Microsoft Teams.

### Built With

The technologies that we used to build Blurr include, but are not limited to:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![WebRTC](https://img.shields.io/badge/webrtc-333333?style=for-the-badge&logo=webrtc&logoColor=white)
![YOLO](https://img.shields.io/badge/yolo-00ADD8?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![React.js](https://img.shields.io/badge/react_js-61DAFB?style=for-the-badge&logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![OCR](https://img.shields.io/badge/tesseract-F44708?style=for-the-badge&logo=tesseract&logoColor=white)
