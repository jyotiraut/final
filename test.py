from fastapi import FastAPI, Query, HTTPException
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, sqrt, atan2
import xgboost as xgb

app = FastAPI()

# Connect to MongoDB
MONGO_URI = "mongodb+srv://anup:1234@cluster0.kuk95.mongodb.net"
client = MongoClient(MONGO_URI)
db = client["Tutor"]

def clean_document(doc):
    """Convert ObjectId fields into strings to make JSON serializable."""
    if isinstance(doc, dict):
        return {key: clean_document(value) for key, value in doc.items()}
    elif isinstance(doc, list):
        return [clean_document(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)  # Convert ObjectId to string
    else:
        return doc  # Return as is

def get_teacher_name(teacher_id):
    """Fetch the teacher's name from the 'users' collection based on teacher_id."""
    try:
        if not isinstance(teacher_id, ObjectId):
            teacher_id = ObjectId(teacher_id)
        user = db["users"].find_one({"_id": teacher_id}, {"fullName": 1})
        if user:
            return user.get("fullName", "No Name Provided")
    except Exception as e:
        print(f"Error fetching teacher name for ID {teacher_id}: {e}")
    return "Unknown"

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
    
   

@app.get("/top-rated-teachers")
def get_top_rated_teachers():
    """Fetches all teachers with an average rating greater than 4.5"""
    top_rated = db["ratings"].find({"averageRating": {"$gt": 4.5}}, {"_id": 1})
    top_teacher_ids = [r["_id"] for r in top_rated]

    if not top_teacher_ids:
        raise HTTPException(status_code=404, detail="No top-rated teachers found")

    teachers = db["teachers"].find({"_id": {"$in": top_teacher_ids}})
    serialized_teachers = []
    for teacher in teachers:
        teacher["fullName"] = get_teacher_name(teacher["_id"])
        serialized_teachers.append(clean_document(teacher))

    return {"top_rated_teachers": serialized_teachers}




@app.get("/recommend-teachers")
def recommend_teachers(query: str = Query(..., description="Search by name, expertise, or address")):
    """Recommend teachers using cosine similarity based on name, expertise, and address."""
    teachers = list(db["teachers"].find({}))

    if not teachers:
        raise HTTPException(status_code=404, detail="No teachers found in database")

    teacher_texts = []
    teacher_ids = []

    for teacher in teachers:
        expertise_str = " ".join(teacher.get("expertise", []))
        teacher_id = teacher["_id"]
        teacher_name = get_teacher_name(teacher_id)
        address_str = teacher.get("contactInfo", {}).get("address", "").lower()
        text_representation = f"{teacher_name.lower()} {expertise_str} {address_str}"
        teacher_texts.append(text_representation)
        teacher_ids.append(teacher_id)

    vectorizer = TfidfVectorizer()
    teacher_vectors = vectorizer.fit_transform(teacher_texts)
    query_vector = vectorizer.transform([query.lower()])

    similarities = cosine_similarity(query_vector, teacher_vectors).flatten()
    similarity_threshold = 0.2

    relevant_teachers = []
    for i, teacher_id in enumerate(teacher_ids):
        if similarities[i] >= similarity_threshold:
            teacher_data = db["teachers"].find_one({"_id": teacher_id})
            if teacher_data:
                teacher_data["fullName"] = get_teacher_name(teacher_id)
                teacher_data["similarity_score"] = float(similarities[i])
                relevant_teachers.append(teacher_data)

    relevant_teachers = sorted(relevant_teachers, key=lambda x: x["similarity_score"], reverse=True)

    if not relevant_teachers:
        return {"message": "No relevant teachers found"}

    return {"recommended_teachers": [clean_document(teacher) for teacher in relevant_teachers]}



@app.get("/ranked-recommendation")
def ranked_recommendation(student_id: str):
    """Recommend teachers based on distance, subject expertise similarity, rating, and XGBoost ranking."""
    student = db["students"].find_one({"_id": ObjectId(student_id)})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Get student details
    student_lat = student.get("location", {}).get("latitude")
    student_lon = student.get("location", {}).get("longitude")
    student_subjects = student.get("subjects", [])

    if not student_lat or not student_lon:
        raise HTTPException(status_code=400, detail="Student location data is missing")
    
    if not student_subjects:
        raise HTTPException(status_code=400, detail="Student has no subjects")

    student_subjects_text = " ".join(student_subjects)  # Convert list to string

    teachers = list(db["teachers"].find({}))
    if not teachers:
        raise HTTPException(status_code=404, detail="No teachers found")

    teacher_features = []
    labels = []

    vectorizer = TfidfVectorizer()  # Initialize outside the loop for efficiency

    for teacher in teachers:
        teacher_location = teacher.get("location", {})
        teacher_lat = teacher_location.get("latitude")
        teacher_lon = teacher_location.get("longitude")
        teacher_expertise = " ".join(teacher.get("expertise", []))  # Convert list to string
        teacher_rating = teacher.get("rating", 0)

        # Skip teachers with missing location data
        if not teacher_lat or not teacher_lon:
            continue  

        # Calculate distance
        distance = haversine(student_lat, student_lon, teacher_lat, teacher_lon)

        # Compute cosine similarity (only if expertise is available)
        if teacher_expertise.strip():
            vectors = vectorizer.fit_transform([student_subjects_text, teacher_expertise])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        else:
            similarity = 0  # Default similarity if teacher has no expertise

        # Store features
        teacher_features.append([distance, similarity, teacher_rating])
        labels.append(str(teacher["_id"]))

    if not teacher_features:
        raise HTTPException(status_code=404, detail="No teachers with valid data found")

    # Convert to NumPy array
    X = np.array(teacher_features)
    y = np.arange(len(teacher_features))  # Ensure correct length

    # Train XGBoost model
    model = xgb.XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)

    # Predict rankings
    predicted_ranks = model.predict(X)

    ranked_teachers = [
        {
            "id": labels[i],
            "fullName": get_teacher_name(labels[i]),
            "distance": teacher_features[i][0],
            "similarity": teacher_features[i][1],
            "rating": teacher_features[i][2],
            "rank": float(predicted_ranks[i])
        }
        for i in range(len(labels))
    ]

    # Sort by predicted rank (higher is better)
    ranked_teachers.sort(key=lambda x: x["rank"], reverse=True)

    return {"ranked_recommendations": ranked_teachers}