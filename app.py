import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Placement Prediction Model")

st.write(
    "Enter your academic and extracurricular details "
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        df = pd.read_csv("placementdata (1).csv"))
    # Select required columns
    df = df[
        [
            "CGPA",
            "Internships",
            "Projects",
            "Workshops/Certifications",
            "ExtracurricularActivities",
            "PlacementTraining",
            "PlacementStatus"
        ]
    ]

    # Convert Yes/No into 1/0
    for col in [
        "ExtracurricularActivities",
        "PlacementTraining"
    ]:
        df[col] = df[col].map({
            "No": 0,
            "Yes": 1
        })

    # Convert placement status into 0/1
    df["PlacementStatus"] = df["PlacementStatus"].map({
        "NotPlaced": 0,
        "Placed": 1
    })

    return df


df = load_data()


# ============================================================
# PREPARE DATA
# ============================================================

X = df.drop(columns=["PlacementStatus"])

y = df["PlacementStatus"]

feature_names = list(X.columns)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# ============================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# USER INPUT SECTION
# ============================================================

st.header("📋 Enter Student Details")

st.write(
    "Please enter your details below:"
)


# CGPA
cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)


# Internships
internships = st.number_input(
    "Number of Internships",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)


# Projects
projects = st.number_input(
    "Number of Projects",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)


# Workshops / Certifications
workshops = st.number_input(
    "Workshops / Certifications Completed",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)


# Extracurricular Activities
extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)


# Placement Training
training = st.selectbox(
    "Placement Training",
    ["Yes", "No"]
)


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Placement Chance",
    use_container_width=True
):

    # Convert Yes / No to 1 / 0
    extracurricular_value = (
        1 if extracurricular == "Yes" else 0
    )

    training_value = (
        1 if training == "Yes" else 0
    )


    # Create DataFrame for student
    student = pd.DataFrame([{

        "CGPA": cgpa,

        "Internships": internships,

        "Projects": projects,

        "Workshops/Certifications": workshops,

        "ExtracurricularActivities":
            extracurricular_value,

        "PlacementTraining":
            training_value

    }])[feature_names]


    # Scale student data
    student_scaled = scaler.transform(student)


    # Predict probability
    probability = model.predict_proba(
        student_scaled
    )[0][1]


    chance = round(
        probability * 100,
        2
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.header("🎯 Prediction Result")

    st.metric(
        "Estimated Placement Probability",
        f"{chance}%"
    )

    st.progress(
        int(chance)
    )


    # Result category
    if chance >= 75:

        st.success(
            "🟢 High Placement Potential"
        )

        st.write(
            "Your profile shows a strong estimated "
            "probability of getting placed."
        )

    elif chance >= 50:

        st.warning(
            "🟡 Moderate Placement Potential"
        )

        st.write(
            "Your profile has a moderate estimated "
            "placement probability. There is room for improvement."
        )

    else:

        st.error(
            "🔴 Low Placement Potential"
        )

        st.write(
            "Your current profile has a lower estimated "
            "placement probability. Consider improving the areas below."
        )


    # ========================================================
    # PERSONALIZED SUGGESTIONS
    # ========================================================

    st.header("💡 Personalized Suggestions")

    suggestions = []


    # Projects
    if projects < 3:

        gap = 3 - projects

        suggestions.append(
            f"📌 Add {gap} more Project(s). "
            f"Placed students typically have 3+ projects, "
            f"while you currently have {projects}."
        )


    # Workshops
    if workshops < 2:

        gap = 2 - workshops

        suggestions.append(
            f"📜 Complete {gap} more Workshop/Certification(s). "
            f"Placed students typically have 2+."
        )


    # Extracurricular
    if extracurricular != "Yes":

        suggestions.append(
            "🏆 Get involved in an Extracurricular Activity. "
            "This showed a positive relationship with placement "
            "in the dataset."
        )


    # Placement Training
    if training != "Yes":

        suggestions.append(
            "🎯 Consider taking Placement Training. "
            "Students with placement training had a higher "
            "placement rate in this dataset."
        )


    # Internship
    if internships < 1:

        suggestions.append(
            "💼 Try to get at least 1 Internship. "
            "However, internships showed a smaller effect "
            "compared with some other factors in this dataset."
        )


    # Display Suggestions
    if suggestions:

        for i, suggestion in enumerate(
            suggestions,
            1
        ):

            st.write(
                f"**{i}.** {suggestion}"
            )

    else:

        st.success(
            "🎉 Your profile already covers "
            "the key factors well!"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ About the Model")

    st.write(
        "This website uses Logistic Regression "
        "to predict student placement probability."
    )

    st.write(
        "**Input Features:**"
    )

    st.write(
        "• CGPA\n"
        "• Internships\n"
        "• Projects\n"
        "• Workshops/Certifications\n"
        "• Extracurricular Activities\n"
        "• Placement Training"
    )

    st.write(
        "**Target:**"
    )

    st.write(
        "Placement Status"
    )

    st.divider()

    st.caption(
        "Machine Learning Project"
    )
