import streamlit as st
from groq import Groq
from PIL import Image
import base64
from io import BytesIO

# ==========================================
# [설정] 페이지 기본 설정 (가장 먼저 실행)
# ==========================================
st.set_page_config(
    page_title="Paper Writer (Multi-Image)", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# [중요] Groq API 키 로드 (Secrets 연동)
# ==========================================
# 이미 Secrets 설정을 완료하셨으므로, 이 코드가 정상 작동할 것입니다.
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("🚨 API 키가 설정되지 않았습니다!")
    st.info("💡 [배포 후] Streamlit Cloud 앱 설정 > Secrets 메뉴에 'GROQ_API_KEY'를 추가해주세요.")
    st.info("💡 [로컬 실행] .streamlit/secrets.toml 파일을 확인해주세요.")
    st.stop()

# ==========================================
# [함수] 이미지 변환
# ==========================================
def encode_image_to_base64(image):
    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ==========================================
# [함수] 자연스러운 논문 생성 로직 (다중 이미지 지원)
# ==========================================
def generate_natural_method(api_key, domain_text, image_list):
    client = Groq(api_key=api_key)
    
    # 1. 텍스트 프롬프트 구성
    user_prompt = f"""
    You are an elite AI researcher writing the **"Proposed Method"** section for a top-tier conference paper (e.g., CVPR, NeurIPS).
    
    **GOAL:** Analyze the attached architecture diagrams (there may be multiple images detailing different parts) and write a **cohesive, logically flowing** description of the proposed framework.
    
    **INSTRUCTIONS:**
    1. **No Artificial Segmentation:** Do NOT force the text into too many sub-sections. Prioritize a **smooth narrative flow**.
    2. **Synthesize Information:** If multiple images are provided (e.g., overall architecture + detailed module), synthesize them into a single coherent explanation.
    3. **Academic Rigor:** Use high-level academic English. Use **LaTeX** for all variables and formulas ($x$, $L_{{total}}$).
    4. **Detail Oriented:** Describe exactly what is happening in the images. Start with the overall pipeline and naturally transition into the details of specific components.

    [Context Info]
    - **Domain:** {domain_text}
    - **Visual Input:** {len(image_list)} architecture diagram(s) attached.

    **Structure Guide:**
    - Start with a strong paragraph summarizing the overall framework.
    - Dedicate substantial paragraphs to detailing the key modules shown in the diagrams.
    - Conclude with the training objectives or inference strategy.
    
    Start writing the "Proposed Method" section now.
    """

    # 2. 메시지 페이로드 구성
    content_payload = [{"type": "text", "text": user_prompt}]

    for img in image_list:
        base64_img = encode_image_to_base64(img)
        content_payload.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}",
            },
        })

    # 3. 모델 ID 설정 (수정됨: 90b -> 11b)
    # ⚠️ Groq에서 90b vision 모델을 내렸으므로 11b로 변경했습니다.
    model_id = "llama-3.2-11b-vision-preview" 

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content_payload,
                }
            ],
            model=model_id, 
            temperature=0.5, 
            max_tokens=6000, 
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Groq 오류 발생: {str(e)}"

# ==========================================
# [UI] 화면 구성
# ==========================================
st.title("📄 AI Paper Writer (Multi-Image Support)")

col1, col2 = st.columns([1, 1])

with col1:
    domain_input = st.text_area(
        "1. 도메인 설명 및 핵심 키워드",
        height=300,
        placeholder="예: Multi-agent debating framework using ViT and LLM..."
    )

with col2:
    uploaded_files = st.file_uploader(
        "2. 아키텍처 이미지 업로드 (여러 장 선택 가능)", 
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"✅ 총 {len(uploaded_files)}장의 이미지가 선택되었습니다.")
        tabs = st.tabs([f"이미지 {i+1}" for i in range(len(uploaded_files))])
        
        pil_images = []
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            pil_images.append(image)
            with tabs[i]:
                st.image(image, caption=uploaded_file.name, use_container_width=True)
    else:
        pil_images = []

st.divider()

if st.button("🚀 자연스러운 논문 작성 시작", type="primary", use_container_width=True):
    if not pil_images:
        st.error("이미지를 한 장 이상 업로드해주세요!")
    else:
        with st.spinner(f'AI가 {len(pil_images)}장의 그림과 설명을 분석하여 글을 쓰고 있습니다...'):
            result = generate_natural_method(GROQ_API_KEY, domain_input, pil_images)
            
            st.divider()
            if "❌" in result:
                st.error(result)
            else:
                st.subheader("📄 생성 결과")
                st.markdown(result)
                st.divider()
                st.text_area("전체 복사 (Ctrl+A)", value=result, height=800)
