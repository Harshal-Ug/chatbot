from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_ollama import OllamaLLM
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from typing import List
from datetime import datetime, timedelta
import jwt

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "service_db"
MENU_COLLECTION = "menuItems"
DOCTOR_COLLECTION = "doctors"
CHAT_COLLECTION_RESTAURANT = "chat_history_restaurant"
CHAT_COLLECTION_DOCTOR = "chat_history_doctor"
USER_COLLECTION = "users"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
menu_collection = db[MENU_COLLECTION]
doctor_collection = db[DOCTOR_COLLECTION]
chat_collection_restaurant = db[CHAT_COLLECTION_RESTAURANT]
chat_collection_doctor = db[CHAT_COLLECTION_DOCTOR]
user_collection = db[USER_COLLECTION]

# LLM model
llm = OllamaLLM(model="gemma2:2b")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT authentication
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# User model
class User(BaseModel):
    email: EmailStr
    password: str


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(authorization: str = Header(None)):
    """Retrieve user from JWT token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing token")

    token = authorization.split("Bearer ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# User authentication routes
@app.post("/signup")
async def signup(user: User):
    existing_user = await user_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(user.password)
    await user_collection.insert_one({"email": user.email, "password": hashed_password})

    return {"message": "User registered successfully."}


@app.post("/login")
async def login(user: User):
    db_user = await user_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(
        data={"sub": db_user["email"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": access_token, "token_type": "bearer"}


async def get_chat_history(user_email: str, mode: str, limit: int = 5) -> List[str]:
    """Retrieve the last few messages to maintain context, based on the mode"""
    chat_collection = chat_collection_restaurant if mode == "restaurant" else chat_collection_doctor
    history = await chat_collection.find({"user_id": user_email}).sort("timestamp", -1).limit(limit).to_list(length=limit)
    return [msg["message"] for msg in history][::-1]


@app.get("/chat")
async def chat(query: str = Query(...), mode: str = Query(...), user_email: str = Depends(get_current_user)):
    """Handles restaurant and doctor inquiries separately using LLM"""
    
    # Ensure mode is valid
    if mode not in ["restaurant", "appointment"]:
        return {"response": "Invalid mode. Please select either 'restaurant' or 'appointment'."}

    chat_history = await get_chat_history(user_email, mode)
    chat_context = "\n".join(chat_history)

    if mode == "restaurant":
        menu_data = await menu_collection.find({}, {"_id": 0}).to_list(length=100)
        if not menu_data:
            return {"response": "Sorry, the menu is currently unavailable."}

        formatted_menu = "\n".join([f"- {item['name']}: ${item['price']}" for category in menu_data for item in category["items"]])

        prompt = f"""
        You are a restaurant inquiry assistant. You only answer questions about the menu.

        Menu Items:
        {formatted_menu}

        Chat History:
        {chat_context}

        User Query: {query}

        Respond clearly and conversationally. Never respond in JSON format.
        """

        chat_collection = chat_collection_restaurant

    elif mode == "appointment":
        doctors = await doctor_collection.find({}, {"_id": 0}).to_list(length=100)
        if not doctors:
            return {"response": "Sorry, no doctor data is available right now."}

        formatted_doctors = "\n".join(
            [f"- Dr. {doctor['name']} ({doctor['specialization']}): Fee ₹{doctor['consultationFee']}."
             for doctor in doctors]
        )

        prompt = f"""
        You are a medical appointment inquiry assistant. You only answer questions about doctors.

        Available Doctors:
        {formatted_doctors}

        Chat History:
        {chat_context}

        User Query: {query}

        Respond in a natural and friendly tone. Never output JSON data.
        """

        chat_collection = chat_collection_doctor

    # Get response from LLM
    ai_response = await llm.ainvoke(prompt)

    # Store chat in the appropriate history collection
    await chat_collection.insert_one({
        "user_id": user_email,
        "message": query,
        "response": ai_response,
        "timestamp": datetime.utcnow()
    })

    return {"response": ai_response}
