import hashlib
import logging
import streamlit as st
from Ats.resume_parser import extract_resume_text
from Ats.ranking import score_resume, rank_resumes, clean_skills
from Ats.matching_chain import analyze_resume
from rag.tools import create_rag_tools
from rag.pipeline import build_rag_vectorstore, ask_rag_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ATS + RAG Resume Screening", page_icon="📄", layout="wide")
st.title("ATS Resume Screening & Ranking")
st.caption("Local AI-powered ATS + RAG | Qwen + Ollama + LangChain")

def reset_rag():
    for key in ["rag_mode", "rag_messages", "rag_vectorstore", "rag_tools"]:
        st.session_state.pop(key, None)

def reset_results():
    reset_rag()
    for key in ["ranked_candidates", "job_title", "job_description", "required_skills", "analysis_fingerprint"]:
        st.session_state.pop(key, None)

def calculate_fingerprint(job_title, job_description, min_years, max_years, skills, qualifications, uploaded_files):
    hasher = hashlib.sha256()
    values = [job_title, job_description, str(min_years), str(max_years), skills, qualifications]
    for value in values:
        hasher.update(str(value).encode("utf-8"))
    for file in uploaded_files or []:
        hasher.update(file.name.encode("utf-8"))
        hasher.update(file.getvalue())
    return hasher.hexdigest()

st.header("1. Job Description")

job_title = st.text_input("Job title", placeholder="e.g. Accountant")
job_description = st.text_area("Full job description", height=200)

col1, col2 = st.columns(2)

with col1:
    min_years = st.number_input("Minimum experience (years)", min_value=0.0, max_value=60.0, value=1.0, step=0.5)

with col2:
    max_years = st.number_input("Maximum experience (years)", min_value=0.0, max_value=60.0, value=3.0, step=0.5)

if min_years > max_years:
    st.warning("Minimum experience exceeds maximum experience.")

required_skills_text = st.text_input(
    "Required skills (comma-separated)",
    placeholder="Accounting, Microsoft Excel, QuickBooks"
)

qualifications = st.text_area(
    "Accepted qualifications (comma-separated)",
    placeholder="Accounting, Finance, Commerce, Business Administration (BBA)"
)

st.caption(
    "Qualifications are treated as alternatives: matching one accepted qualification satisfies the qualification criterion."
)

skills = clean_skills([
    item.strip()
    for item in required_skills_text.split(",")
    if item.strip()
])

with st.expander("Preview parsed skills"):
    if skills:
        st.write(", ".join(skills))
    else:
        st.write("No skills entered.")

st.header("2. Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload candidate resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    help="Upload PDF or DOCX files up to 20 MB each."
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} resume(s) uploaded.")

current_fingerprint = calculate_fingerprint(
    job_title,
    job_description,
    min_years,
    max_years,
    required_skills_text,
    qualifications,
    uploaded_files
)

previous_fingerprint = st.session_state.get("analysis_fingerprint")

if previous_fingerprint and previous_fingerprint != current_fingerprint:
    reset_results()
    st.info("Inputs have changed. Run the analysis again to update the rankings.")

analyze_clicked = st.button(
    "Analyze & Rank Resumes",
    type="primary",
    use_container_width=True
)

if analyze_clicked:
    if not job_title.strip():
        st.error("Please enter the job title.")
    elif not job_description.strip():
        st.error("Please enter the job description.")
    elif min_years > max_years:
        st.error("Minimum experience cannot exceed maximum experience.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    elif len({file.name.lower() for file in uploaded_files}) != len(uploaded_files):
        st.error("Duplicate filenames detected. Please rename the files uniquely.")
    else:
        reset_results()
        candidates = []
        errors = []
        progress = st.progress(0)
        status = st.empty()
        total = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):
            status.write(f"Processing {index + 1}/{total}: {uploaded_file.name}")

            try:
                resume_text = extract_resume_text(uploaded_file)

                analysis = analyze_resume(
                    job_title=job_title,
                    job_description=job_description,
                    required_skills=skills,
                    min_years=min_years,
                    max_years=max_years,
                    qualifications=qualifications,
                    resume_text=resume_text
                )

                result = score_resume(
                    resume_text=resume_text,
                    required_skills=skills,
                    min_years=min_years,
                    max_years=max_years,
                    qualifications=qualifications,
                    job_description=job_description,
                    analysis=analysis,
                    job_title=job_title
                )

                candidates.append({
                    "filename": uploaded_file.name,
                    "resume_text": resume_text,
                    **result
                })

            except Exception as exc:
                logger.exception("Resume processing failed: %s", uploaded_file.name)
                errors.append(f"{uploaded_file.name}: {exc}")

            progress.progress((index + 1) / total)

        status.empty()
        progress.empty()

        if candidates:
            ranked_candidates = rank_resumes(candidates)
            st.session_state["ranked_candidates"] = ranked_candidates
            st.session_state["job_title"] = job_title
            st.session_state["job_description"] = job_description
            st.session_state["required_skills"] = skills
            st.session_state["analysis_fingerprint"] = current_fingerprint
            st.success(f"Successfully ranked {len(ranked_candidates)} candidates.")
        else:
            st.error("No resumes could be processed.")

        for error in errors:
            st.warning(error)

if "ranked_candidates" in st.session_state:
    st.divider()
    st.header("3. Ranked Candidates")

    ranked_candidates = st.session_state["ranked_candidates"]
    col1, col2, col3 = st.columns(3)
    top = ranked_candidates[0]

    col1.metric("Total Candidates", len(ranked_candidates))
    col2.metric("Top Candidate", top["filename"])
    col3.metric("Top ATS Score", f"{top['score']:.2f}%")

    st.warning(
        "ATS scores are preliminary matching indicators. They should support, not replace, human review."
    )

    for candidate in ranked_candidates:
        analysis = candidate.get("analysis", {})

        with st.expander(
            f"Rank {candidate['rank']} | {candidate['filename']} | {candidate['score']:.2f}%",
            expanded=candidate["rank"] == 1
        ):
            st.metric("Overall ATS Match", f"{candidate['score']:.2f}%")

            score_cols = st.columns(4)
            score_cols[0].metric("Skills", f"{candidate['skill_score']:.1f}%")
            score_cols[1].metric("Experience", f"{candidate['experience_score']:.1f}%")
            score_cols[2].metric("Qualifications", f"{candidate['qualification_score']:.1f}%")
            score_cols[3].metric("JD Overlap", f"{candidate['description_score']:.1f}%")

            st.divider()
            st.subheader("Skills Assessment")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Matched Skills**")
                matched = candidate.get("matched_skills", [])

                if matched:
                    for skill in matched:
                        st.write(f"✅ {skill}")
                else:
                    st.write("No positive matches identified.")

            with col2:
                st.write("**Skills Without Positive Evidence**")
                missing = candidate.get("missing_skills", [])

                if missing:
                    for skill in missing:
                        st.write(f"⚠️ {skill}")
                else:
                    st.write("All listed skills matched.")

            st.divider()
            st.subheader("Experience Assessment")

            total_experience = candidate.get("experience_years")
            relevant_experience = candidate.get("relevant_experience_years")

            st.write(
                "**Total documented experience:**",
                f"{total_experience:g} years" if total_experience is not None else "Not verified"
            )

            st.write(
                "**Relevant dated experience:**",
                f"{relevant_experience:g} years" if relevant_experience is not None else "Not verified"
            )

            st.write("**Assessment:**", candidate.get("experience_status", "Unknown"))

            summary = analysis.get("experience_summary", "")
            if summary:
                st.caption(summary)

            st.divider()
            st.subheader("Qualifications")

            matched_qualifications = candidate.get("matched_qualifications", [])

            if matched_qualifications:
                for qualification in matched_qualifications:
                    st.write(f"✅ {qualification}")
            else:
                st.write("No accepted qualification identified.")

            education = candidate.get("education_evidence", [])
            for item in education:
                st.write(f"Education: {item}")

            st.divider()
            st.subheader("Relevant Projects")

            projects = analysis.get("relevant_projects", [])

            if projects:
                for project in projects:
                    st.write(f"**{project.get('name', '')}**")
                    st.write(project.get("relevance", ""))
                    st.caption(project.get("evidence", ""))
            else:
                st.write("No verified relevant projects identified.")

            st.divider()
            st.subheader("Matching Evidence")

            for evidence in analysis.get("matching_evidence", []):
                st.write(f"- {evidence}")

            st.divider()
            st.subheader("Requirements Needing Verification")

            for requirement in analysis.get("missing_requirements", []):
                st.write(f"- {requirement}")

            with st.expander("View Extracted Resume Text"):
                st.text(candidate["resume_text"])

if "ranked_candidates" in st.session_state:
    st.divider()
    st.header("4. Resume RAG Assistant")

    st.write(
        "Ask questions about candidates, compare resumes, or investigate job requirements."
    )

    if st.button("Continue to RAG", type="primary", use_container_width=True):
        reset_rag()
        st.session_state["rag_mode"] = True
        st.session_state["rag_messages"] = []
        st.rerun()

    if st.session_state.get("rag_mode", False):
        if "rag_tools" not in st.session_state:
            with st.spinner("Building resume knowledge base..."):
                try:
                    vectorstore = build_rag_vectorstore(
                        candidates=st.session_state["ranked_candidates"],
                        job_title=st.session_state["job_title"],
                        job_description=st.session_state["job_description"]
                    )

                    tools = create_rag_tools(
                        vectorstore=vectorstore,
                        candidates=st.session_state["ranked_candidates"],
                        job_title=st.session_state["job_title"],
                        job_description=st.session_state["job_description"]
                    )

                    st.session_state["rag_vectorstore"] = vectorstore
                    st.session_state["rag_tools"] = tools

                except Exception:
                    logger.exception("RAG initialization failed.")
                    st.error(
                        "Failed to initialize the RAG assistant. Verify that Ollama and the embedding model are available."
                    )

        if "rag_tools" in st.session_state:
            st.subheader("Ask Your Resume Assistant")

            st.caption(
                "Answers use stored ATS results and retrieved document evidence."
            )

            messages = st.session_state.get("rag_messages", [])

            for message in messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            question = st.chat_input("Ask about candidates or job requirements...")

            if question:
                st.session_state["rag_messages"].append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving and verifying evidence..."):
                        try:
                            answer = ask_rag_question(
                                question=question,
                                history=st.session_state["rag_messages"][:-1],
                                tools=st.session_state["rag_tools"]
                            )
                        except Exception:
                            logger.exception("RAG question failed.")
                            answer = "The assistant could not answer this question reliably."

                        st.markdown(answer)

                st.session_state["rag_messages"].append({
                    "role": "assistant",
                    "content": answer
                })

            if st.button("Clear Conversation"):
                st.session_state["rag_messages"] = []
                st.rerun()

st.divider()
st.caption("ATS + RAG | Streamlit | Python | LangChain | Ollama | Qwen 3.5 0.8B")