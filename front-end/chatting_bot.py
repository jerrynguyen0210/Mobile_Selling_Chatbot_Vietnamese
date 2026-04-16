import os
import uuid
import streamlit as st
import requests
import time

# --- Page Config ---
st.set_page_config(
    page_title="Mobile Shop Chatbot",
    page_icon="📱",
    layout="centered",
)

# --- Styling ---
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f0f2f6;
    }

    /* Chat message bubbles */
    .user-bubble {
        background-color: #0084ff;
        color: white;
        padding: 10px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 4px 0;
        max-width: 75%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .bot-bubble {
        background-color: #ffffff;
        color: #1a1a1a;
        padding: 10px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 4px 0;
        max-width: 75%;
        margin-right: auto;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    /* Row wrappers */
    .user-row {
        display: flex;
        justify-content: flex-end;
        margin: 6px 0;
    }
    .bot-row {
        display: flex;
        justify-content: flex-start;
        margin: 6px 0;
        align-items: flex-end;
        gap: 8px;
    }

    /* Bot avatar */
    .bot-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: #0084ff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }

    /* Timestamp */
    .msg-time {
        font-size: 11px;
        color: #999;
        text-align: right;
        margin-top: 2px;
        padding-right: 4px;
    }

    /* Product card */
    .product-card {
        background-color: #f8f9ff;
        border: 1px solid #e0e4f0;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        max-width: 75%;
        font-size: 13px;
    }
    .product-card .product-name {
        font-weight: 600;
        color: #0084ff;
        margin-bottom: 4px;
    }
    .product-card .product-meta {
        color: #555;
    }

    /* Divider */
    hr { border: none; border-top: 1px solid #e0e0e0; margin: 8px 0; }

    /* Sidebar background & base text */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: white;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stCaption {
        color: #ffffff !important;
    }

    /* Sidebar buttons: solid visible background, strong text */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #2e2e4a !important;
        color: #ffffff !important;
        border: 1px solid #4a4a72 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        transition: background-color 0.15s, border-color 0.15s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #3d3d6b !important;
        border-color: #6e6eaa !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton > button:active {
        background-color: #0084ff !important;
        border-color: #0084ff !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Backend URL (from environment variable, not hardcoded) ---
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def check_backend_health() -> bool:
    """Return True if backend liveness probe responds OK."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/health/live", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def create_backend_session() -> str:
    """Create a new chat session on the backend and return the session_id UUID string."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/chat/session",
            json={},
            timeout=5,
        )
        if resp.status_code == 201:
            return str(resp.json()["session_id"])
    except Exception:
        pass
    # Backend unreachable — generate a local UUID so the format stays valid
    return str(uuid.uuid4())


def search_products(query: str, brands: list[str], price_min_vnd: int, price_max_vnd: int) -> list[dict]:
    """Search products via backend API. Returns list of ProductCard dicts."""
    try:
        payload: dict = {
            "query": query,
            "brands": brands,
            "price_range": {
                "min": price_min_vnd,
                "max": price_max_vnd,
            },
            "in_stock_only": False,
            "page": 1,
            "page_size": 5,
        }
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/products/search",
            json=payload,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except Exception:
        pass
    return []


def get_bot_response(user_message: str, session_id: str) -> tuple[str, list[dict]]:
    """
    Call ``POST /api/v1/chat/message`` and return (content, source_documents).
    Automatically renews the session if it has expired (404).
    Falls back to a local reply when the backend is unreachable.
    """
    for attempt in range(2):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/v1/chat/message",
                json={"session_id": session_id, "message": user_message},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.backend_online = True
                return data.get("content", ""), data.get("source_documents", [])
            if resp.status_code == 404 and attempt == 0:
                # Session expired — create a new one and retry
                st.session_state.session_id = create_backend_session()
                session_id = st.session_state.session_id
                continue
        except requests.exceptions.ConnectionError:
            st.session_state.backend_online = False
            break
        except Exception:
            break

    # --- Fallback: local keyword replies (backend unreachable) ---
    msg_lower = user_message.lower()
    if any(k in msg_lower for k in ["iphone", "apple"]):
        return (
            "🍎 **iPhone** là lựa chọn tuyệt vời! Hiện tại chúng tôi có:\n"
            "- iPhone 15 Pro Max: 34.990.000đ\n"
            "- iPhone 15: 22.990.000đ\n"
            "- iPhone 14: 19.990.000đ\n\n"
            "Bạn quan tâm đến mẫu nào?",
            [],
        )
    if any(k in msg_lower for k in ["samsung", "galaxy"]):
        return (
            "📱 **Samsung Galaxy** series:\n"
            "- Galaxy S24 Ultra: 31.990.000đ\n"
            "- Galaxy A55: 10.990.000đ\n"
            "- Galaxy A35: 7.990.000đ\n\n"
            "Bạn muốn xem thêm thông tin về model nào?",
            [],
        )
    if any(k in msg_lower for k in ["xiaomi", "redmi", "poco"]):
        return (
            "🔥 **Xiaomi** - Hiệu năng cao, giá tốt:\n"
            "- Xiaomi 14: 18.990.000đ\n"
            "- Redmi Note 13 Pro: 7.490.000đ\n"
            "- POCO X6 Pro: 8.990.000đ",
            [],
        )
    if any(k in msg_lower for k in ["5 triệu", "5tr", "dưới 5", "rẻ"]):
        return (
            "💰 Điện thoại **dưới 5 triệu** tốt nhất:\n"
            "1. Xiaomi Redmi 13C - 3.290.000đ\n"
            "2. Samsung Galaxy A15 - 4.490.000đ\n"
            "3. OPPO A18 - 3.790.000đ\n\n"
            "Tất cả đều có bảo hành 12 tháng chính hãng!",
            [],
        )
    if any(k in msg_lower for k in ["pin", "trâu", "lâu"]):
        return (
            "🔋 Điện thoại **pin trâu** nhất hiện nay:\n"
            "1. Xiaomi Poco C65 - Pin 5.000mAh\n"
            "2. Samsung Galaxy M55 - Pin 6.000mAh\n"
            "3. Infinix Note 30 - Pin 5.000mAh + Sạc 45W",
            [],
        )
    if any(k in msg_lower for k in ["chụp ảnh", "camera", "ảnh đẹp"]):
        return (
            "📸 Điện thoại **camera tốt nhất**:\n"
            "1. iPhone 15 Pro - Camera 48MP, LiDAR\n"
            "2. Samsung S24 Ultra - Camera 200MP\n"
            "3. Google Pixel 8 - AI Camera tốt nhất\n\n"
            "Bạn ưu tiên selfie hay ảnh phong cảnh?",
            [],
        )
    return (
        f"Cảm ơn bạn đã hỏi về **\"{user_message}\"**. Tôi có thể tư vấn cho bạn về:\n"
        "- 📱 Các dòng điện thoại theo hãng\n"
        "- 💰 Điện thoại theo tầm giá\n"
        "- 📸 Camera, pin, hiệu năng\n\n"
        "Bạn cần tư vấn thêm điều gì?",
        [],
    )


# ---------------------------------------------------------------------------
# Session State initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! 👋 Tôi là trợ lý tư vấn điện thoại. Bạn cần tìm kiếm dòng máy nào? Tôi có thể giúp bạn tìm điện thoại phù hợp với nhu cầu và ngân sách.",
            "time": time.strftime("%H:%M"),
            "source_documents": [],
        }
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = create_backend_session()

if "backend_online" not in st.session_state:
    st.session_state.backend_online = check_backend_health()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# Category → (display label, brand filter for product search, chat query)
CATEGORIES = [
    ("📱 iPhone",  ["Apple"],   "Điện thoại iPhone"),
    ("🤖 Android", [],          "Điện thoại Android tốt nhất"),
    ("💻 Samsung", ["Samsung"], "Điện thoại Samsung"),
    ("🔥 Xiaomi",  ["Xiaomi"],  "Điện thoại Xiaomi"),
    ("💡 OPPO",    ["OPPO"],    "Điện thoại OPPO"),
    ("🎮 Gaming",  [],          "Điện thoại gaming hiệu năng cao"),
]

with st.sidebar:
    st.markdown("## 📱 Mobile Shop")
    st.markdown("---")
    st.markdown("### Danh mục phổ biến")
    for label, brands, query in CATEGORIES:
        if st.button(label, use_container_width=True, key=f"cat_{label}"):
            st.session_state.pending_input = query
            st.session_state.pending_brands = brands

    st.markdown("---")
    st.markdown("### Tầm giá")
    price_range = st.select_slider(
        "Ngân sách (triệu VND)",
        options=[2, 3, 5, 7, 10, 15, 20, 30, 50],
        value=(5, 20),
        label_visibility="collapsed",
    )
    st.caption(f"Từ {price_range[0]}tr → {price_range[1]}tr VND")

    st.markdown("---")
    if st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Cuộc trò chuyện đã được xóa. Tôi có thể giúp gì cho bạn?",
                "time": time.strftime("%H:%M"),
                "source_documents": [],
            }
        ]
        # Start a fresh session on the backend
        st.session_state.session_id = create_backend_session()
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("# 📱")
with col2:
    st.markdown("## Tư vấn mua điện thoại")
    status_icon = "🟢" if st.session_state.backend_online else "🔴"
    status_text = "Đang hoạt động · Trả lời ngay lập tức" if st.session_state.backend_online else "Mất kết nối · Đang dùng chế độ offline"
    st.caption(f"{status_icon} {status_text}")

st.markdown("---")


# ---------------------------------------------------------------------------
# Chat Display
# ---------------------------------------------------------------------------

def _fmt_price(vnd: float) -> str:
    """Format a VND price as e.g. 12.990.000đ."""
    return f"{int(vnd):,}đ".replace(",", ".")


def _render_source_documents(docs: list[dict]) -> None:
    """Render product cards returned as source documents from the backend."""
    if not docs:
        return
    for doc in docs:
        name = doc.get("product_name") or doc.get("name", "")
        score = doc.get("score")
        price = doc.get("price")
        brand = doc.get("brand", "")
        specs = doc.get("specs") or {}
        in_stock = doc.get("in_stock", True)

        price_str = f" · {_fmt_price(price)}" if price else ""
        score_str = f" · Độ phù hợp: {score:.0%}" if score else ""
        stock_str = " · Còn hàng ✓" if in_stock else " · Hết hàng"
        ram = specs.get("ram_gb")
        storage = specs.get("storage_gb")
        spec_str = ""
        if ram or storage:
            spec_str = f"RAM {ram}GB / {storage}GB" if ram and storage else (f"RAM {ram}GB" if ram else f"{storage}GB")

        st.markdown(
            f"""
            <div class="product-card">
                <div class="product-name">📦 {name}</div>
                <div class="product-meta">{brand}{price_str}{stock_str}{score_str}</div>
                {"<div class='product-meta'>" + spec_str + "</div>" if spec_str else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )


chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="user-row">
                    <div>
                        <div class="user-bubble">{msg["content"]}</div>
                        <div class="msg-time">{msg.get("time", "")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="bot-row">
                    <div class="bot-avatar">🤖</div>
                    <div>
                        <div class="bot-bubble">{msg["content"]}</div>
                        <div class="msg-time">{msg.get("time", "")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_source_documents(msg.get("source_documents", []))

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Quick Reply Suggestions
# ---------------------------------------------------------------------------

SUGGESTIONS = [
    ("Điện thoại dưới 5 triệu", []),
    ("iPhone mới nhất", ["Apple"]),
    ("Pin trâu nhất", []),
    ("Chụp ảnh đẹp nhất", []),
]

st.markdown("**Gợi ý:**")
cols = st.columns(len(SUGGESTIONS))
for i, (suggestion, brands) in enumerate(SUGGESTIONS):
    with cols[i]:
        if st.button(suggestion, use_container_width=True, key=f"sug_{i}"):
            st.session_state.pending_input = suggestion
            st.session_state.pending_brands = brands

st.markdown("---")


# ---------------------------------------------------------------------------
# Input Area
# ---------------------------------------------------------------------------

with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Nhắn tin",
            placeholder="Nhập tin nhắn của bạn...",
            label_visibility="collapsed",
            key="user_input_field",
        )
    with col_btn:
        submitted = st.form_submit_button("Gửi ➤", use_container_width=True)

# Handle pending input from sidebar / suggestion buttons
pending_brands: list[str] = []
if "pending_input" in st.session_state and st.session_state.pending_input:
    user_input = st.session_state.pending_input
    pending_brands = st.session_state.pop("pending_brands", [])
    submitted = True
    del st.session_state.pending_input


# ---------------------------------------------------------------------------
# Handle Submission
# ---------------------------------------------------------------------------

if submitted and user_input and user_input.strip():
    user_msg = user_input.strip()
    current_time = time.strftime("%H:%M")

    st.session_state.messages.append(
        {"role": "user", "content": user_msg, "time": current_time, "source_documents": []}
    )

    with st.spinner("Đang trả lời..."):
        bot_content, source_docs = get_bot_response(user_msg, st.session_state.session_id)

        # If the chat response has no source documents but we have brand / price
        # context from the sidebar, fetch products from the search API directly.
        if not source_docs and (pending_brands or price_range):
            price_min_vnd = price_range[0] * 1_000_000
            price_max_vnd = price_range[1] * 1_000_000
            source_docs = search_products(user_msg, pending_brands, price_min_vnd, price_max_vnd)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_content,
            "time": time.strftime("%H:%M"),
            "source_documents": source_docs,
        }
    )

    st.rerun()
