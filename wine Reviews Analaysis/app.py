from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "wine_review_secret_key"

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub("[^a-zA-Z]", " ", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in set(stopwords.words("english"))]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    # Remove descriptions that are too short
    if len(text.split()) < 3:
        return ""
    # Join words back into a single string
    return " ".join(words)

# Initialize a flag to ensure the model is loaded only once
model_loaded = False

@app.before_request
def load_model():
    global model, vectorizer, mean_points, model_loaded
    if not model_loaded:
        # If model and vectorizer files exist, load them
        if os.path.exists('model/wine_model.pkl') and os.path.exists('model/vectorizer.pkl'):
            model = pickle.load(open('model/wine_model.pkl', 'rb'))
            vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
            mean_points = pickle.load(open('model/mean_points.pkl', 'rb'))
        else:
            # Load wine dataset
            df = pd.read_csv('data/winemag-data-130k-v2.csv')
            
            # Drop unnamed column
            if 'Unnamed: 0' in df.columns:
                df.drop(['Unnamed: 0'], axis=1, inplace=True)
                
            # Calculate mean points
            mean_points = df.points.mean()
            
            # Create target variable (above average or not)
            df['Above_Average'] = [1 if i > mean_points else 0 for i in df.points]
            
            # Preprocess descriptions with progress indicator
            print("Preprocessing descriptions...")
            description_list = []
            total_descriptions = len(df.description)
            for index, description in enumerate(df.description):
                processed = preprocess_text(description)
                if processed:  # Skip empty descriptions
                    description_list.append(processed)
                # Print progress percentage
                if (index + 1) % 100 == 0 or (index + 1) == total_descriptions:
                    progress = ((index + 1) / total_descriptions) * 100
                    print(f"Preprocessing progress: {progress:.2f}%")
            
            # Use TfidfVectorizer instead of CountVectorizer
            print("Creating vectorizer...")
            vectorizer = TfidfVectorizer(max_features=1500)
            X = vectorizer.fit_transform(description_list).toarray()
            y = df.Above_Average.values

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a RandomForestClassifier
            print("Training model...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            print("Model Accuracy:", accuracy_score(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))
            
            # Create directory for model if it doesn't exist
            if not os.path.exists('model'):
                os.makedirs('model')
                
            # Save model and vectorizer
            pickle.dump(model, open('model/wine_model.pkl', 'wb'))
            pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))
            pickle.dump(mean_points, open('model/mean_points.pkl', 'wb'))
            
        print("Model loaded successfully!")
        model_loaded = True

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Function to check if text is related to wine
def is_wine_related(text):
    wine_keywords = ["wine", "flavor", "aroma", "taste", "grape", "vineyard", "oak", "tannin", "blend", "vintage", 
                     "palate", "bouquet", "finish", "notes", "acidity", "fruity", "dry", "sweet", "spicy"]
    text = text.lower()
    return any(keyword in text for keyword in wine_keywords)

# Get prediction scores for a description
def get_prediction_details(description):
    # Check if the description is related to wine
    if not is_wine_related(description):
        return {
            "description": description,
            "prediction": None,
            "rating_text": "Not a wine-related description",
            "rating_class": "warning",
            "confidence": 0,
            "is_valid": False
        }
    
    # Preprocess description
    processed_text = preprocess_text(description)
    
    # Vectorize description
    description_vector = vectorizer.transform([processed_text]).toarray()
    
    # Make prediction
    prediction = model.predict(description_vector)[0]
    
    # Get probability scores for confidence level
    prediction_proba = model.predict_proba(description_vector)[0]
    confidence = max(prediction_proba) * 100
    
    # Extract key wine features
    features = extract_wine_features(description)
    
    # Determine rating and class based on prediction
    if prediction == 1:
        rating_class = "above-average"
        rating_text = "Good Wine"
        score = prediction_proba[1] * 100
    else:
        rating_class = "below-average"
        rating_text = "Below Average Wine"
        score = prediction_proba[0] * 100
    
    return {
        "description": description,
        "prediction": prediction,
        "rating_text": rating_text,
        "rating_class": rating_class,
        "confidence": confidence,
        "score": score,
        "is_valid": True,
        "features": features
    }

# Function to extract key wine features for recommendation
def extract_wine_features(description):
    text = description.lower()
    
    # Define feature categories and their related keywords
    feature_categories = {
        "body": ["full-bodied", "medium-bodied", "light-bodied", "rich", "heavy", "light"],
        "acidity": ["high acidity", "balanced acidity", "crisp", "bright", "sharp"],
        "sweetness": ["dry", "semi-dry", "sweet", "off-dry"],
        "tannins": ["smooth tannins", "firm tannins", "silky tannins", "structured"],
        "fruit_notes": ["berry", "cherry", "apple", "pear", "citrus", "tropical", "stone fruit", 
                      "plum", "blackberry", "raspberry", "peach"],
        "other_notes": ["oak", "vanilla", "spice", "chocolate", "floral", "herbal", "earthy", 
                       "mineral", "leather", "tobacco", "pepper"]
    }
    
    # Extract features
    features = {}
    for category, keywords in feature_categories.items():
        features[category] = [keyword for keyword in keywords if keyword in text]
    
    return features

# Generate detailed recommendation based on comparison
def generate_recommendation(result1, result2, better_wine):
    if better_wine == 1:
        better = result1
        other = result2
    else:
        better = result2
        other = result1
    
    # If both wines are good, create a detailed recommendation
    if better["prediction"] == 1 and other["prediction"] == 1:
        # Compare confidence scores
        confidence_diff = abs(better["score"] - other["score"])
        
        if confidence_diff < 5:
            strength = "slightly"
        elif confidence_diff < 15:
            strength = "notably"
        else:
            strength = "significantly"
        
        # Create recommendation text
        recommendation = f"Both wines appear to be good choices, but Wine #{better_wine} is better "
        recommendation += f"compared to other%. "
        
        # Add feature-based recommendation if available
        better_features = []
        for category, features in better["features"].items():
            if features:
                if category == "body":
                    better_features.append(f"{features[0]} body")
                elif category == "acidity":
                    better_features.append(f"{features[0]}")
                elif category == "sweetness":
                    better_features.append(f"{features[0]}")
                elif category == "tannins":
                    better_features.append(f"{features[0]}")
                elif category == "fruit_notes" and len(features) > 0:
                    better_features.append(f"{features[0]} notes")
                elif category == "other_notes" and len(features) > 0:
                    better_features.append(f"{features[0]} character")
        
        if better_features:
            recommendation += f"Wine #{better_wine} stands out for its {', '.join(better_features[:3])}."
        
        return recommendation
    elif better["prediction"] == 1:
        return f"Wine #{better_wine} is clearly the better choice with a positive rating, while the other wine is below average."
    else:
        return f"While neither wine is rated as above average, Wine #{better_wine} is the better choice between the two."

# Compare two descriptions route
@app.route('/compare', methods=['POST'])
def compare():
    if request.method == 'POST':
        # Get form data
        description1 = request.form['description1']
        description2 = request.form['description2']
        
        # Check if descriptions are the same (ignoring case and leading/trailing spaces)
        if description1.strip().lower() == description2.strip().lower():
            flash("You've entered the same description for both wines. Please enter different descriptions for comparison.")
            return redirect(url_for('home'))
        
        # Get predictions for both descriptions
        result1 = get_prediction_details(description1)
        result2 = get_prediction_details(description2)
        
        # Determine which wine is better (always pick one, no ties)
        better_wine = None
        if result1["is_valid"] and result2["is_valid"]:
            # First compare by prediction class
            if result1["prediction"] > result2["prediction"]:
                better_wine = 1
            elif result2["prediction"] > result1["prediction"]:
                better_wine = 2
            else:
                # If both have same prediction class, compare confidence score
                if result1["score"] > result2["score"]:
                    better_wine = 1
                else:
                    # For true ties or if wine 2 has higher score, pick wine 2
                    # This ensures we always pick a winner
                    better_wine = 2
        elif result1["is_valid"]:
            better_wine = 1
        elif result2["is_valid"]:
            better_wine = 2
        
        # Generate detailed recommendation for good wines
        recommendation = None
        if better_wine is not None and result1["is_valid"] and result2["is_valid"]:
            recommendation = generate_recommendation(result1, result2, better_wine)
        
        return render_template('compare_result.html', 
                              result1=result1,
                              result2=result2,
                              better_wine=better_wine,
                              recommendation=recommendation)

# Single prediction route (keeping for backward compatibility)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        description = request.form['description']
        result = get_prediction_details(description)
        
        return render_template('result.html', 
                              description=result["description"],
                              prediction=result["prediction"],
                              rating_text=result["rating_text"],
                              rating_class=result["rating_class"],
                              confidence=result["confidence"])

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)