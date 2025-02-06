# Databricks notebook source
# MAGIC %md # Initialization

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

companies = spark.read.parquet('/dbfs/linkedin_train_data')
display(companies)

# COMMAND ----------

profiles = spark.read.parquet('/dbfs/linkedin_people_train_data')
#profiles_og = spark.read.parquet('/dbfs/linkedin_people_train_data')
display(profiles)

# COMMAND ----------

# ESSAI
# 
# 

'''from pyspark.sql.functions import col

# Filter rows where 'about_context' and 'recommendations_context' are not null
profiles = profiles.filter(
    (col("about").isNotNull()) & (col("recommendations").isNotNull())
)

# Sample 
profiles = profiles.limit(5)'''

from pyspark.sql.functions import col, size

# Assuming your DataFrame is called 'df'
profiles = profiles.filter((col("about").isNotNull()) & (size(col("recommendations")) > 0)).limit(10)

# Display the filtered DataFrame
display(profiles)

# COMMAND ----------

# ---------------- POSTER EXAMPLE ----------------

from pyspark.sql.functions import col
profiles = profiles.filter(col("id") == "carayou")
display(profiles)

# COMMAND ----------

from pyspark.sql.functions import col, when, regexp_extract, explode, udf
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md # PROFILES PART

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                                        PROFILES
------------------------------------------------------------------------------------------------'''

# COMMAND ----------

# MAGIC %md # top_university 

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        TOP UNIVERSITY FEATURE IN PROFILES DATAFRAME
    -------------------------------------------------------------------------------------------- '''

from pyspark.sql.functions import col, when, lower

# List of US top 20 universities
universities_names = [
    "princeton", "stanford", "massachusetts institute of technology", "yale",
    "berkeley", "columbia", "university of pennsylvania", "harvard", "rice",
    "cornell", "northwestern", "johns hopkins", "university of california",
    "university of chicago", "vanderbilt", "dartmouth", "williams", "brown",
    "claremont mckenna", "duke"
]

# Create a regex pattern for matching any of the university names (case insensitive)
regex_pattern = "|".join([fr"\b{university.lower()}\b" for university in universities_names])

# Add a boolean feature indicating whether 'educations_details' matches the pattern
profiles = profiles.withColumn(
    "top_university",
    when(lower(col("educations_details")).rlike(regex_pattern), 1).otherwise(0)
)

display(profiles)

# COMMAND ----------

# MAGIC %md # degrees

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        DEGREES FEATURE IN PROFILES DATAFRAME 
    -------------------------------------------------------------------------------------------- '''
    
from pyspark.sql.functions import col, explode, collect_list, array_distinct, lower
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.sql.window import Window

# 1) Define the standardization function
def extract_standard_degree(deg_str: str) -> str:
    if deg_str is None:
        return None

    deg_str = deg_str.lower()

    # Bachelor's Degree Variations
    bachelor_variations = ["bachelor", "bsc", "b.a.", "b.a", "b.s.", "b.s", "bachelor of science", "licence"]
    if any(bach in deg_str for bach in bachelor_variations):
        return "Bachelor"

    # Master's Degree Variations
    master_variations = ["master", "msc", "m.s.", "m.s", "m.a.", "m.a", "master of science", "master of arts"]
    if any(mast in deg_str for mast in master_variations):
        return "Master"

    # Doctorate/Ph.D. Variations
    doctorate_variations = ["phd", "ph.d", "doctorate", "doctoral", "dr.", "doctor"]
    if any(doc in deg_str for doc in doctorate_variations):
        return "Doctorate"

    # Associate's Degree Variations
    associate_variations = ["associate", "a.a.", "a.a", "a.s.", "a.s", "assoc."]
    if any(assoc in deg_str for assoc in associate_variations):
        return "Associate"

    return None  # Exclude unrecognized degrees

# 2) Create UDF for Spark
extract_standard_degree_udf = udf(extract_standard_degree, StringType())

# 3) Explode, apply UDF, and filter recognized degrees
profiles = profiles.withColumn("education_exploded", explode(col("education")))

# Apply UDF and filter for valid degrees
profiles = profiles.withColumn(
    "standardized_degree",
    extract_standard_degree_udf(lower(col("education_exploded.degree")))
)

# 4) Use Window Function to Collect Degrees Without Aggregation
window_spec = Window.partitionBy("name")
profiles = profiles.withColumn(
    "degrees",
    array_distinct(collect_list("standardized_degree").over(window_spec))
)

# Drop intermediate columns if desired
profiles = profiles.drop("education_exploded", "standardized_degree")

profiles = profiles.dropDuplicates(["id"])


# COMMAND ----------

# MAGIC %md # volunteering

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        VOLUNTEERING FEATURE IN PROFILES DATAFRAME
    -------------------------------------------------------------------------------------------- '''

from pyspark.sql.functions import size, col, when, explode, udf, lit, create_map
from pyspark.sql.types import MapType, StringType, ArrayType
from pyspark.sql import functions as F


# Add the 'volunteering' column: 1 if 'volunteer_experience' is non-empty, otherwise 0
profiles = profiles.withColumn(
    "volunteering",
    when(size(col("volunteer_experience")) > 0, 1).otherwise(0)
)

# Créer une nouvelle colonne 'cause_volunteer' avec des causes uniques
profiles = profiles.withColumn(
    "cause_volunteer",
    F.expr("array_distinct(transform(volunteer_experience, x -> x['cause']))")
)

# ADD FROM VOLUNTEERING + FROM VOLUNTEERING CAUSES

# Define a mapping function for 'cause_volunteer'
def map_volunteering_to_values(causes):
    values_dict = {}
    volunteering_common_values = ['Community', 'Contribution', 'Meaningful work']

    for value in volunteering_common_values:
        # Ensure the key exists in the dictionary and initialize it as a list if needed
        if value not in values_dict:
            values_dict[value] = [['volunteering', None],]

    if not causes:  # Handle None or empty lists
        return values_dict

    for cause in causes:
        if cause == 'Education':
            if 'Altruism' in values_dict:
                values_dict['Altruism'].append(['volunteering', cause])
            else : 
                values_dict['Altruism'] = [['volunteering', cause],]

        elif cause in ['Human Rights', 'Civil Rights and Social Action']:
            if 'Equality' in values_dict:
                values_dict['Equality'].append(['volunteering', cause])
            else : 
                values_dict['Equality'] = [['volunteering', cause],]

        elif cause == 'Arts and Culture':
            if 'Creativity' in values_dict:
                values_dict['Creativity'].append(['volunteering', cause])
            else : 
                values_dict['Creativity'] = [['volunteering', cause],]

        elif cause in ['Health', 'Disaster and Humanitarian Relief', 'Animal Welfare', 'Poverty Alleviation']:
            if 'Empathy' in values_dict:
                values_dict['Empathy'].append(['volunteering', cause])
            else : 
                values_dict['Empathy'] = [['volunteering', cause],]

        elif cause == 'Disaster and Humanitarian Relief':
            if 'Empathy' in values_dict:
                values_dict['Empathy'].append(['volunteering', cause])
            else : 
                values_dict['Empathy'] = [['volunteering', cause],]

        elif cause == 'Social Services':
            if 'Equity' in values_dict:
                values_dict['Equity'].append(['volunteering', cause])
            else : 
                values_dict['Equity'] = [['volunteering', cause],]

        elif cause == 'Animal Welfare':
            if 'Empathy' in values_dict:
                values_dict['Empathy'].append(['volunteering', cause])
            else : 
                values_dict['Empathy'] = [['volunteering', cause],]

        elif cause == 'Poverty Alleviation':
            if 'Empathy' in values_dict:
                values_dict['Empathy'].append(['volunteering', cause])
            else : 
                values_dict['Empathy'] = [['volunteering', cause],]

        elif cause == 'Environment':
            if 'Environment' in values_dict:
                values_dict['Environment'].append(['volunteering', cause])
            else:
                values_dict['Environment'] = [['volunteering', cause],]

        elif cause == 'Children':
            if 'Altruism' in values_dict:
                values_dict['Altruism'].append(['volunteering', cause])
            else : 
                values_dict['Altruism'] = [['volunteering', cause],]

    return values_dict

# Create an empty map using create_map()
empty_map = create_map()

# Register the UDF
map_volunteering_udf = udf(map_volunteering_to_values, MapType(StringType(), ArrayType(ArrayType(StringType()))))

# Add the new column with a conditional application
profiles = profiles.withColumn(
    'values_from_volunteering',
    when(
        size(col('volunteer_experience')) > 0,  # Check if the size of volunteer_experience > 0
        map_volunteering_udf(col('cause_volunteer'))  # Apply the UDF
    ).otherwise(lit(None).cast(MapType(StringType(), ArrayType(ArrayType(StringType())))))  # Use null for others
)

# Display the updated DataFrame
display(profiles)

# COMMAND ----------

# MAGIC %md # profile picture

# COMMAND ----------


''' --------------------------------------------------------------------------------------------
                        PROFILE PICTURE FEATURE IN PROFILES DATAFRAME
    -------------------------------------------------------------------------------------------- '''
    
!pip install ultralytics 
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Load YOLO model for face detection
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(model_path)

# Load emotion detection pipeline
emotion_pipe = pipeline("image-classification", model="jayanta/microsoft-resnet-50-cartoon-emotion-detection")

# Define the processing function
def process_profile_picture(image_url):
    try:
        # Download the image
        response = requests.get(image_url)
        if response.status_code != 200:
            return "0"  # Return "0" if image can't be fetched

        image = Image.open(BytesIO(response.content))

        # Face detection
        face_output = face_model(image)
        if len(face_output[0].boxes) == 0:
            return "0"  # No face detected

        # Emotion detection
        emotions = emotion_pipe(image)
        return [emotions]  # Convert to string for PySpark compatibility

    except Exception as e:
        print(f"Error processing URL {image_url}: {e}")
        return "0"  # Return "0" for any errors

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType

# Define the UDF
process_profile_picture_udf = udf(process_profile_picture, ArrayType(StringType()))

# Apply the UDF to the 'avatar' column and create a new column 'avatar_emotions'
profiles = profiles.withColumn("avatar_emotions", process_profile_picture_udf(F.col("avatar")))

# Display the updated DataFrame
display(profiles)


# COMMAND ----------

# MAGIC %md # average post duration

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        AVERAGE POST DURATION FEATURE IN PROFILES DATAFRAME
    -------------------------------------------------------------------------------------------- '''

from pyspark.sql.functions import col, when, regexp_extract, explode, udf
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F

# Define a UDF to convert "duration_short" to total months
def duration_to_months(duration_short):
    if duration_short is None:
        return None
    import re
    match = re.match(r'(?:(\d+)\s*years?)?\s*(?:(\d+)\s*months?)?', duration_short)
    if not match:
        return None
    years = int(match.group(1)) if match.group(1) else 0
    months = int(match.group(2)) if match.group(2) else 0
    return years * 12 + months

# Register the UDF
duration_to_months_udf = F.udf(duration_to_months, returnType=IntegerType())

# Explode the "experience" array to process each element individually
exploded_df = profiles.withColumn("experience_exploded", F.explode("experience"))

# Add a column for the total months for each experience
exploded_df = exploded_df.withColumn(
    "months",
    duration_to_months_udf(F.col("experience_exploded.duration_short"))
)

# Group by original row and compute the average months (ignoring nulls) and round to one decimal place
average_months_df = exploded_df.groupBy("id").agg(
    F.round(F.avg("months"), 1).alias("average_months_of_experience")
)

# Join the result back to the original DataFrame
profiles = profiles.join(average_months_df, on="id", how="left")

display(profiles)


# COMMAND ----------

# MAGIC %md # organization Type

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                    ORGANIZATION TYPE FROM EXPERIENCE FEATURE IN PROFILES DATAFRAME --- 
    -------------------------------------------------------------------------------------------- '''


from pyspark.sql.functions import collect_list, explode, array_distinct, array_union, coalesce, col, lit,struct, size


# Step 1: Group profiles by company name and collect employee IDs
companies_with_employees = (
    profiles.groupBy("current_company:name")
    .agg(collect_list("id").alias("employee_ids"))
)

# Step 2: Extract experience details for each profile
profiles_pushed = profiles.select('id', 'experience')
exploded_profiles = profiles_pushed.withColumn("experience_exploded", explode("experience"))

# Step 3: Extract 'id' and 'subtitle' from the exploded experience column
profiles_with_subtitle = exploded_profiles.select(
    "id", col("experience_exploded.subtitle").alias("subtitle")
)

# Step 4: Group by 'id' and collect all 'subtitle' values into a list
grouped_profiles = profiles_with_subtitle.groupBy("id").agg(
    collect_list("subtitle").alias("subtitles")
)

# Step 5: Explode subtitles to map subtitles to IDs
exploded_subtitles = grouped_profiles.withColumn("subtitle", explode("subtitles")).select('id', 'subtitle')

# Step 6: Group by subtitle to collect IDs
subtitles_with_ids = exploded_subtitles.groupBy("subtitle").agg(
    collect_list("id").alias("ids")
)

# Step 7: Perform a full outer join between companies_with_employees and subtitles_with_ids
updated_companies_with_employees = companies_with_employees.join(
    subtitles_with_ids,
    companies_with_employees["current_company:name"] == subtitles_with_ids["subtitle"],
    "full_outer"
)

# Step 8: Merge the 'employee_ids' and 'ids' columns, ensuring unique values
companies_employees = updated_companies_with_employees.withColumn(
    "employee_ids",
    array_union(
        coalesce(col("employee_ids"), lit([])),  # Replace null with an empty list
        coalesce(col("ids"), lit([]))  # Replace null with an empty list
    )
).select(
    coalesce(col("current_company:name"), col("subtitle")).alias("company_name"),
    "employee_ids"
)

# Step 9: Add organization_type information from the companies DataFrame
# Only keep companies present in both DataFrames
companies_employees_filtered = companies_employees.join(
    companies,
    companies_employees["company_name"] == companies["name"],
    "inner"  # Inner join to keep only matched companies
).select(
    "company_name", "employee_ids", "organization_type"
)

# Step 10: Explode employee_ids to assign organization types to each ID
exploded_employees = companies_employees_filtered.withColumn("id", explode("employee_ids"))

# Step 11: Group by 'id' and collect all organization types worked in
types_per_id = exploded_employees.groupBy("id").agg(
    array_distinct(
        collect_list(
            struct(col("company_name"), col("organization_type"))
        )
    ).alias("company_and_organization_types")
)

profiles= profiles.join(types_per_id, on='id', how='left')


display(profiles)




# COMMAND ----------

# MAGIC %md # profile values

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
        VALUES FEATURE ON PROFILES DATAFRAME WITH SOURCES INFERED DIRECTLY FROM FEATURES
    -------------------------------------------------------------------------------------------- '''
    
# ------------------------------- ADD FROM OTHER FEATURES -------------------------------

profiles = profiles.withColumn("values_from_features", F.struct())

# ADD FROM TOP_UNIVERSITY
top_university_features = ['Excellence', 'Learning']

profiles = profiles.withColumn(
    "values_from_features",
    F.col("values_from_features").withField(
        "top_university",
        F.when(
            F.col("top_university") == 1,
            F.array(*[F.lit(x) for x in top_university_features])
        ).otherwise(F.lit(None).cast("array<string>"))  # ensure consistent type
    )
)

# ADD FROM DEGREES
doctorate_features = ['Excellence', 'Autonomy', 'Learning']

profiles = profiles.withColumn(
    "values_from_features",
    F.col("values_from_features").withField(
        "degrees",
        F.when(
            F.array_contains(F.col("degrees"), "Doctorate"),
            F.array(*[F.lit(x) for x in doctorate_features])
        ).otherwise(F.lit(None).cast("array<string>"))
    )
)



# ADD FROM ORGANIZATION_TYPE
nonprofit_features = ['Contribution', 'Meaningful work', 'Community']
self_employed_features = ['Accountability', 'Autonomy']

profiles = profiles.withColumn(
    "values_from_features",
    F.col("values_from_features").withField(
        "company_and_organization_types",
        F.when(
            F.expr("array_contains(transform(company_and_organization_types, x -> x['organization_type']), 'Nonprofit')"),
            F.array(*[F.lit(x) for x in nonprofit_features])
        )
        .when(
            F.expr("array_contains(transform(company_and_organization_types, x -> x['organization_type']), 'Self-Employed')"),
            F.array(*[F.lit(x) for x in self_employed_features])
        )
        .otherwise(F.lit(None).cast("array<string>"))
    )
)


# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        EXTRACTED KEYWORDS FEATURED IN PROFILES DATAFRAME --- PART 1
    -------------------------------------------------------------------------------------------- '''


profiles_pushed_for_keywords = profiles.select('id', 'about', 'recommendations')


# FROM 'ABOUT' SECTION 

# Extract KEYWORDS

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import nltk

# Step 1: Define UDF with NLTK Resource Download and Filters
def extract_keywords_nltk(text):
    if text is None :
        return None
    # Ensure NLTK resources are available on executors
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk import pos_tag

    # Tokenize and tag POS
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Extract nouns and adjectives, lowercase, and filter for letters and hyphens
    stop_words = set(stopwords.words('english'))
    keywords = [
        word.lower() for word, pos in pos_tags
        if (pos.startswith('NN') or pos.startswith('JJ'))  # Only nouns and adjectives
        and word.lower() not in stop_words  # Remove stopwords
        and all(c.isalpha() or c == '-' for c in word)  # Allow only letters and hyphens
    ]

    # Return distinct keywords
    return list(set(keywords))

# Step 2: Register UDF
nltk_udf = udf(extract_keywords_nltk, ArrayType(StringType()))

# Step 4: Apply UDF to Add 'about_keywords' Column
profiles_pushed_for_keywords = profiles_pushed_for_keywords.withColumn("about_keywords", nltk_udf(profiles_pushed_for_keywords["about"]))

# FROM RECOMMENDATIONS SECTION 

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import nltk

# Step 1: Define UDF for Recommendations
def extract_keywords_from_list_nltk(recommendations):
    if not recommendations:
        return []
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk import pos_tag

    # Ensure input is a valid list
    if not isinstance(recommendations, list):
        return []

    # Define stopwords
    stop_words = set(stopwords.words('english'))

    # Initialize an empty set for unique keywords
    keywords_set = set()

    # Process each recommendation in the list
    for text in recommendations:
        if not isinstance(text, str) or not text.strip():  # Skip non-string or empty values
            continue
        try:
            # Tokenize and tag POS
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            # Extract nouns and adjectives, lowercase, and filter for letters and hyphens
            keywords = [
                word.lower() for word, pos in pos_tags
                if (pos.startswith('NN') or pos.startswith('JJ'))  # Only nouns and adjectives
                and word.lower() not in stop_words  # Remove stopwords
                and all(c.isalpha() or c == '-' for c in word)  # Allow only letters and hyphens
            ]

            # Add keywords to the set
            keywords_set.update(keywords)
        except Exception as e:
            # Log or handle unexpected errors for specific strings
            print(f"Error processing text: {text}, error: {e}")
            continue

    # Return distinct keywords as a list
    return list(keywords_set)

# Step 2: Register UDF for Recommendations
recommendations_udf = udf(extract_keywords_from_list_nltk, ArrayType(StringType()))

# Step 3: Apply UDF to Add 'recommendations_keywords' Column
profiles_pushed_for_keywords = profiles_pushed_for_keywords.withColumn("recommendations_keywords", recommendations_udf(profiles_pushed_for_keywords["recommendations"]))

display(profiles_pushed_for_keywords)


# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        VALUES FEATURE ON PROFILES DATAFRAME ---- PART 2
    -------------------------------------------------------------------------------------------- '''

# ------------------------------- MODEL BASED - WITH CONTEXT -------------------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

# Define the label columns and threshold
THRESHOLD = 0.6
LABEL_COLUMNS = ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
                 'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal',
                 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
                 'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
                 'Universalism: objectivity']

# Define a function to apply the model to a single string
def apply_model_to_text(text):
    if not text:
        return None

    # Load the model and tokenizer inside the function
    tokenizer = AutoTokenizer.from_pretrained("tum-nlp/Deberta_Human_Value_Detector")
    model = AutoModelForSequenceClassification.from_pretrained("tum-nlp/Deberta_Human_Value_Detector", trust_remote_code=True)

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        prediction = model(encoding["input_ids"], encoding["attention_mask"])
        prediction = prediction["output"].flatten().numpy()

    # Filter labels based on threshold
    results = []
    for label, score in zip(LABEL_COLUMNS, prediction):
        if score >= THRESHOLD:
            results.append(f"{label}: {score:.4f}")
    return results


# FROM ABOUT SECTION 
# Define a UDF for the function
@udf(ArrayType(StringType()))
def apply_model_udf(text):
    if not text :
        return []
    result = apply_model_to_text(text)
    # Extract the first word before the first colon
    extracted_words = [item.split(':')[0] for item in result]
    return extracted_words

# Apply the UDF to the 'about' column and create a new column 'about_context'
profiles_pushed_for_keywords = profiles_pushed_for_keywords.withColumn("about_context", apply_model_udf(profiles_pushed_for_keywords["about"]))


# FROM RECOMMENDATIONS SECTION 
# Define a UDF to handle lists of recommendations
@udf(returnType=ArrayType(StringType()))
def process_recommendations(recommendations):
    if not recommendations:  # Handles both None and empty list
        return None
    results = []
    for rec in recommendations:
        results.extend(apply_model_to_text(rec))
    # Extract the first word before the first colon
    extracted_words = [item.split(':')[0] for item in results]
    return extracted_words

# Apply the UDF to the DataFrame
profiles_pushed_for_keywords = profiles_pushed_for_keywords.withColumn("recommendations_context", process_recommendations(profiles_pushed_for_keywords["recommendations"]))

display(profiles_pushed_for_keywords)


# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        VALUES FEATURE ON PROFILES DATAFRAME ---- PART 3
    -------------------------------------------------------------------------------------------- '''

from pyspark.sql import SparkSession
from sparknlp.base import *
from sparknlp.annotator import *

def get_values_dict():
    ''' --------------------------------------------------------------------------------------------
                                        VALUES DICTIONARY
    -------------------------------------------------------------------------------------------- '''
    
    core_values_lexical_field = {
    "Accountability": [
        "responsibility", 
        "responsible"
        "ownership", 
        "reliability"
    ],
    "Adaptability": [
        "flexible", 
        "resilient", 
        "versatile", 
        "adaptive"
    ],
    "Achievement": [
        "accomplishment", 
        "success", 
        "victory", 
        "performance",
        "productivity"
    ],
    "Passion": [
        'passion',
        'dedication',
        'enthousiasm',
        'commitment'
    ]
    "Empathy": [
        "compassion", 
        "kindness", 
        "benevolence", 
        "charity",
        "empathy"
    ],
    "Collaboration": [
        "cooperation", 
        "teamwork", 
        "partnership", 
        "collaborative",
        "collaboration",
        "synergy"
    ],
    "Communication": [
        "dialogue", 
        "interaction", 
        "exchanging",
        "communicate",
    ],
    "Community": [
        "society", 
        "collective", 
        "network"
    ],

    "Creativity": [
        "originality",
        "creativity",
        "artist"
    ],

    "Altruism":[
        "generosity",
        "altruism",
        "nonprofit"
    ],

    "Contribution": [
        "donation", 
        "engagement", 
        "participation", 
        "involvement"
    ],
    "Innovation": [
        "creative", 
        "ingenuity", 
        "inventive",
        "innovation",
        "visionary"
    ],
    "Diversity": [
        "inclusion", 
        "heterogeneity"
        "diversity",
        "multiculturalism"
    ],
    "Environment": [
        "sustainability", 
        "eco",
        "environment"
        "ecology",
        "eco-friendly", 
        "long-term viability", 
        "conservation", 
        "renewability",
        "green",
        "sustainability"
    ],
    "Equality": [
        "equity", 
        "fairness", 
        "impartiality", 
        "evenness"
    ],
    "Growth": [
        "development", 
        "progress", 
        "advancement", 
        "expansion"
    ],
    "Work Ethic": [
        "professionalism",
        "ethic",
        "diligence"
    ],
    "Integrity": [
        "honesty", 
        "authenticity", 
        "fair", 
        "sincere", 
        "openness"
    ],
    "Leadership": [
        "guidance", 
        "directive", 
        "influent", 
        "authority"
    ],
    "Learning": [
        "knowledge", 
        "training",
        "self-improvement"
    ],
    "Perseverance": [
        "determination", 
        "tenacity", 
        "resolve", 
        "steadfastness",
    ],
    "Respect": [
        "admiration", 
        "regard", 
        "deference", 
        "esteem"
    ],
    "Balance": [
        "stability", 
        "equilibrium", 
        "wellness", 
        "balance"
    ],
    "High Standards": [
        "quality",
        "excellence",
        "expertise"
    ],

    'Meaningful work': [
        "meaningful",
    ],

    "Autonomy": [
        "autonomy",
        "independant",
        "self driven"
    ]
    }

    def add_stemmed_and_lemmatized_words_to_dictionary(core_values_lexical_field):
        """
        Adds stemmed and lemmatized words to the list of values for each key in the dictionary.
        """
        # Convert dictionary into a DataFrame for processing
        key_value_pairs = [(key, word) for key, values in core_values_lexical_field.items() for word in values]
        data = spark.createDataFrame(key_value_pairs, ["key", "word"])
    
        # Define the Spark NLP pipeline for stemming and lemmatization
        document_assembler = DocumentAssembler() \
        .setInputCol("word") \
        .setOutputCol("document")

        tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

        stemmer = Stemmer() \
        .setInputCols(["token"]) \
        .setOutputCol("stem")

        lemmatizer = LemmatizerModel.pretrained() \
        .setInputCols(["token"]) \
        .setOutputCol("lemma")

        pipeline = Pipeline(stages=[
        document_assembler,
        tokenizer,
        stemmer,
        lemmatizer
        ])

        # Apply the pipeline to the data
        model = pipeline.fit(data)
        result = model.transform(data)

        # Extract original words, stems, lemmas, and keys
        result = result.selectExpr("key", "word", "explode(stem.result) as stemmed", "explode(lemma.result) as lemmatized")
    
        # Convert the DataFrame back into a dictionary with stemmed and lemmatized words added
        result_rdd = result.rdd.map(lambda row: (row['key'], row['word'], row['stemmed'], row['lemmatized']))
        updated_dict = {}

        for key, word, stemmed, lemmatized in result_rdd.collect():
            if key not in updated_dict:
                updated_dict[key] = set(core_values_lexical_field[key])  # Add original words
            updated_dict[key].add(stemmed)  # Add the stemmed word
            updated_dict[key].add(lemmatized)  # Add the lemmatized word

        # Convert sets back to sorted lists
        for key in updated_dict:
            updated_dict[key] = sorted(updated_dict[key])

        return updated_dict
    
    # Add stemmed and lemmatized words to the dictionary
    core_values_lexical_field_with_stemmed_and_lemmatized = add_stemmed_and_lemmatized_words_to_dictionary(core_values_lexical_field)

    return core_values_lexical_field_with_stemmed_and_lemmatized



# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        VALUES FEATURE ON PROFILES DATAFRAME ---- PART 4
    -------------------------------------------------------------------------------------------- '''
    
# -------------------- GATHER ALL DETECTED KEYWORDS AND MODEL-BASED VALUES  -------------------

from pyspark.sql import functions as F
columns = ['about_context', 'recommendations_context', 'about_keywords', 'recommendations_keywords']

'''# Create the `keywords_agg` column
profiles_pushed_for_keywords = profiles_pushed_for_keywords.withColumn(
    'keywords_agg',
    F.array_distinct(F.flatten(F.array('about_context', 'recommendations_context', 'about_keywords', 'recommendations_keywords')))
)'''

# Create the `keywords_agg` column
profiles_pushed_for_keywords = profiles_pushed_for_keywords.withColumn(
    'keywords_agg',
    (F.flatten(F.array('about_keywords', 'recommendations_keywords')))
)
display(profiles_pushed_for_keywords)

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                        VALUES FEATURE ON PROFILES DATAFRAME ---- PART 5 --- VERSION 1
    -------------------------------------------------------------------------------------------- '''
    

dict_values = get_values_dict()

@udf(returnType=ArrayType(StringType()))
def filter_map_udf(keywords):
    """
    A PySpark UDF that takes a list of keywords (ArrayType(StringType))
    and returns a list of matched core values.
    """
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

    # Ensure resources are downloaded on workers
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    
    lemmatizer = WordNetLemmatizer()
    
    def lemmatize_word(word):
        if not word:
            return ""
        return lemmatizer.lemmatize(word.lower(), pos="n")
    
    def filter_and_map_keywords(keywords, core_values_lexical_field):
        if len(keywords)==0:
            return []
        matched_values = []
        for kw in keywords:
            kw_lemma = lemmatize_word(kw)
            for core_val, synonyms_set in core_values_lexical_field.items():
                if kw_lemma in synonyms_set:
                    matched_values.append(core_val)
                    break
        return matched_values

    if not keywords:
        return []
    
    # Retrieve broadcasted dictionary
    local_dict = bc_dict_values.value
    
    # Filter and map keywords
    return filter_and_map_keywords(keywords, local_dict)


# Apply the UDF
profiles_pushed_for_keywords = (
    profiles_pushed_for_keywords
    .withColumn("profile_values_not_sourced", filter_map_udf(col("keywords_agg")))
)

display(profiles_pushed_for_keywords)

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
VALUES FEATURE ON PROFILES DATAFRAME ---- TO DO PART 5 --- VERSION 2 with WORD EMBEDDINGS Extension
    -------------------------------------------------------------------------------------------- '''

import sparknlp
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, FloatType
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, Normalizer, WordEmbeddingsModel
from pyspark.ml import Pipeline
from scipy.spatial.distance import cosine
import numpy as np
from collections import defaultdict

# Initialize Spark NLP
spark = sparknlp.start()

# Invert the dictionary: synonym -> set of core values
def invert_core_values_dict(core_values_dict):
    inverted = defaultdict(set)
    for cv, synonyms in core_values_dict.items():
        for syn in synonyms:
            inverted[syn.lower()].add(cv)  # Ensure lowercase for matching
    return dict(inverted)

inverted_dict = invert_core_values_dict(core_values_lexical_field)

# Prepare Synonym DataFrame
synonym_rows = [(synonym, list(core_values)) for synonym, core_values in inverted_dict.items()]
synonyms_df = spark.createDataFrame(synonym_rows, ["text", "core_values"])

# Define Spark NLP Pipeline
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized") \
    .setLowercase(True)

embeddings = WordEmbeddingsModel.pretrained("glove_100d") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    normalizer,
    embeddings
])

# Fit the pipeline on the synonyms DataFrame
nlp_model = nlp_pipeline.fit(synonyms_df)
synonyms_with_embeddings = nlp_model.transform(synonyms_df)

# Collect Synonym Embeddings into a Python Dictionary
synonym_embeddings_map = {}
for row in synonyms_with_embeddings.select("text", "core_values", "embeddings").collect():
    if row.embeddings:
        embedding_vector = row.embeddings[0].embeddings  # First token embedding
        synonym_embeddings_map[row.text] = (embedding_vector, set(row.core_values))

# Broadcast the Synonym Embeddings
bc_synonym_embeddings_map = spark.sparkContext.broadcast(synonym_embeddings_map)

# Define UDF to Find Core Values Using Cosine Similarity
def semantic_core_values_finder(sentence_embedding):
    """
    Match sentence embedding to synonym embeddings via cosine similarity.
    """
    if not sentence_embedding:
        return []

    matched_values = set()
    synonym_map = bc_synonym_embeddings_map.value  # Access the broadcasted dictionary

    sentence_vector = np.array(sentence_embedding, dtype="float32")

    threshold = 0.85 # Threshold for matching HYPER PARAMETER

    # Compare with each synonym embedding
    for synonym, (syn_vector, core_values) in synonym_map.items():
        syn_vector = np.array(syn_vector, dtype="float32")
        similarity = 1 - cosine(sentence_vector, syn_vector)

        if similarity > threshold:  # Threshold for matching
            matched_values.update(core_values)
    threshold = 0.85 # Threshold for matching HYPER PARAMETER

    # Compare with each synonym embedding
    for synonym, (syn_vector, core_values) in synonym_map.items():
        syn_vector = np.array(syn_vector, dtype="float32")
        similarity = 1 - cosine(sentence_vector, syn_vector)

        if similarity > threshold:  # Threshold for matching
            matched_values.update(core_values)
    return list(matched_values)

semantic_core_values_udf = F.udf(semantic_core_values_finder, ArrayType(StringType()))

# Join keywords into a single text column for processing
profiles = profiles.withColumn("text", F.concat_ws(" ", "keywords_agg"))

# Apply the NLP pipeline to the profiles DataFrame
profiles_with_embeddings = nlp_model.transform(profiles)

# Aggregate Token Embeddings to Sentence-Level Embedding (Mean Embedding)
def mean_embeddings(embeddings):
    if embeddings:
        vectors = np.array([e.embeddings for e in embeddings])
        return vectors.mean(axis=0).tolist()
    return None

mean_embeddings_udf = F.udf(mean_embeddings, ArrayType(FloatType()))
profiles_with_embeddings = profiles_with_embeddings.withColumn(
    "sentence_embedding",
    mean_embeddings_udf(F.col("embeddings"))
)

# Detect Core Values Using the UDF
profiles_with_embeddings = profiles_with_embeddings.withColumn(
    "detected_core_values",
    semantic_core_values_udf(F.col("sentence_embedding"))
)

# Display Results
profiles_with_embeddings.select("keywords_agg", "detected_core_values").show(truncate=False)


# COMMAND ----------

''' --------------------------------------------------------------------------------------------
            JOINING VALUES FEATURE ON PUSHED PROFILES DATAFRAME WITH PROFILES ---- PART 6 
    -------------------------------------------------------------------------------------------- '''

# Perform the join
profiles = profiles.join(
    profiles_pushed_for_keywords.select("id", "profile_values_not_sourced"),  # Select only the required columns
    on="id",  # Join on the 'id' column
    how="inner"  # You can change the join type if needed (e.g., 'left', 'right', 'outer')
)

# COMMAND ----------

''' --------------------------------------------------------------------------------------------
                                        ENGINEERED PROFILES
    -------------------------------------------------------------------------------------------- '''

display(profiles)
profiles_engineered = profiles
profiles_engineered.write.format("parquet").mode("overwrite").save("/mnt/lab94290/results/parquet-data/profiles_engineered_ejt")



# COMMAND ----------

# Load the Parquet file
profiles_engineered = spark.read.parquet("/mnt/lab94290/results/parquet-data/profiles_engineered_ejt")

# Display the DataFrame
display(profiles_engineered)

# COMMAND ----------

# MAGIC %md # COMPANIES PART

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                                            COMPANIES
------------------------------------------------------------------------------------------------'''

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                    COMPANIES WITH AT LEAST 20 EMPLOYEES_ID DATAFRAME
------------------------------------------------------------------------------------------------'''

from pyspark.sql.functions import collect_list, explode, array_distinct, array_union, coalesce, col, lit, count


# Assuming 'profiles' is your DataFrame
companies_with_employees = (
    profiles.groupBy("current_company:name")
    .agg(collect_list("id").alias("employee_ids"))
)

profiles_pushed = profiles.select('id', 'experience')

# Step 1: Explode the 'experience' column to get each experience as a separate row
exploded_profiles = profiles_pushed.withColumn("experience_exploded", explode("experience"))

# Step 2: Extract 'id' and 'subtitle' from the exploded experience column
profiles_with_subtitle = exploded_profiles.select(
    "id", col("experience_exploded.subtitle").alias("subtitle")
)

# Step 3: Group by 'id' and collect all 'subtitle' values into a list
grouped_profiles = profiles_with_subtitle.groupBy("id").agg(
    collect_list("subtitle").alias("subtitles")
)

# Step 1: Explode the 'subtitles' column in 'grouped_profiles'
exploded_subtitles = grouped_profiles.withColumn("subtitle", explode("subtitles")).select('id', 'subtitle')

# Group by 'subtitle' and collect all 'id' values into a list
subtitles_with_ids = exploded_subtitles.groupBy("subtitle").agg(
    collect_list("id").alias("ids")
)


# Perform a full outer join between companies_with_employees and subtitles_with_ids
updated_companies_with_employees = companies_with_employees.join(
    subtitles_with_ids,
    companies_with_employees["current_company:name"] == subtitles_with_ids["subtitle"],
    "full_outer"
)

# Merge the 'employee_ids' and 'ids' columns, ensuring unique values
companies_employees = updated_companies_with_employees.withColumn(
    "employee_ids",
    array_union(
        coalesce(col("employee_ids"), lit([])),  # Replace null with an empty list
        coalesce(col("ids"), lit([]))  # Replace null with an empty list
    )
).select(
    coalesce(col("current_company:name"), col("subtitle")).alias("company_name"),
    "employee_ids"
)

from pyspark.sql.functions import array_distinct, size, col

# Ensure 'employee_ids' contains only distinct values
companies_employees = companies_employees.withColumn(
    "employee_ids", array_distinct(col("employee_ids"))
)

# Add a column for the size of the 'employee_ids' list
companies_employees = companies_employees.withColumn(
    "employee_count", size(col("employee_ids"))
)

# Filter companies with at least 20 employees
companies_employees = companies_employees.filter(
    col("employee_count") >= 20
).drop("employee_count")

# Drop rows where 'company_name' is null
companies_employees = companies_employees.filter(col("company_name").isNotNull())

# Only keep companies present in both DataFrames
companies_employees = companies_employees.join(
    companies,
    companies_employees["company_name"] == companies["name"],
    "inner"  # Inner join to keep only matched companies
).select("company_name", "employee_ids")

# Find duplicated company names by grouping and counting
duplicated_companies = companies_employees.groupBy("company_name").agg(
    count("company_name").alias("count")
).filter(col("count") > 1)

# Join with the original DataFrame to display the rows with duplicated company names
duplicated_rows = companies_employees.join(
    duplicated_companies, on="company_name", how="inner"
)

# Create a DataFrame with distinct company_name values from duplicated_rows
distinct_duplicated_companies = duplicated_rows.select("company_name").distinct()

# Perform a left anti-join to drop rows where 'company_name' is in distinct_duplicated_companies
companies_employees = companies_employees.join(
    distinct_duplicated_companies, on="company_name", how="left_anti"
)

display(companies_employees)



# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                                        COMPANY PATTERNS
------------------------------------------------------------------------------------------------'''

from pyspark.sql.functions import explode, col, avg, variance
from pyspark.sql.functions import regexp_extract, col, concat_ws
from pyspark.sql.functions import (
    explode, explode_outer, regexp_extract, when, col, expr, udf, concat_ws, avg, variance, round as spark_round
)
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType

from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType

from pyspark.sql import Window

# Explode the 'employee_ids' column in the 'companies_employees' DataFrame
exploded_df = companies_employees.withColumn("employee_id", explode(col("employee_ids"))).drop("employee_ids")


# Join the exploded DataFrame with the 'profiles' DataFrame
patterns_df = exploded_df.join(profiles, exploded_df["employee_id"] == profiles["id"], "inner")

# Define a window that partitions by 'company_name'
window_spec = Window.partitionBy("company_name")


# -------- ON VOLUNTEERING --------

patterns_df = patterns_df.withColumn(
    "volunteering_percentage",
    avg(col("volunteering")).over(window_spec) * 100
)

# -------- ON TOP UNIVERSITIES --------

patterns_df = patterns_df.withColumn(
    "top_university_percentage",
    avg(col("top_university")).over(window_spec) * 100
)

# -------- ON AVERAGE POST DURATIONS --------

patterns_df = patterns_df.withColumn(
    "mean_post_duration",
    avg(col("average_months_of_experience")).over(window_spec)
)

patterns_df = patterns_df.withColumn(
    "variance_post_duration",
    variance(col("average_months_of_experience")).over(window_spec)
)

# -------- ON DEGREES --------

# Explode the 'degree' column into individual degrees
patterns_df = patterns_df.withColumn("degree", explode_outer(col("degrees")))


# Calculate the total number of employees per company
patterns_df = patterns_df.withColumn(
    "total_employees", count("degree").over(window_spec)
)

# Calculate the count of each degree type per company
patterns_df = patterns_df.withColumn(
    "bachelor_count",
    count(when(col("degree") == "Bachelor", 1)).over(window_spec),
).withColumn(
    "associate_count",
    count(when(col("degree") == "Associate", 1)).over(window_spec),
).withColumn(
    "master_count",
    count(when(col("degree") == "Master", 1)).over(window_spec),
).withColumn(
    "no_degree_count",
    count(when(col("degree").isNull() | (col("degree") == ""), 1)).over(window_spec),
)

# Calculate the degree percentages and add it as a new column
patterns_df = patterns_df.withColumn(
    "degree_percentage",
    concat_ws(
        ", ",
        concat_ws("", lit("Bachelor: "), spark_round((col("bachelor_count") * 100.0 / col("total_employees")), 2), lit(" %")),
        concat_ws("", lit("Associate: "), spark_round((col("associate_count") * 100.0 / col("total_employees")), 2), lit(" %")),
        concat_ws("", lit("Master: "), spark_round((col("master_count") * 100.0 / col("total_employees")), 2), lit(" %")),
        concat_ws("", lit("No Degree: "), spark_round((col("no_degree_count") * 100.0 / col("total_employees")), 2), lit(" %"))
    )
)

# Drop unnecessary intermediate columns (optional)
patterns_df = patterns_df.drop("bachelor_count", "associate_count", "master_count", "no_degree_count", "total_employees")
patterns_df = patterns_df.dropDuplicates(["id"])


'''
# -------- ON PROFILE PICTURE EMOTIONS --------

# Expressions régulières pour extraire les scores
happy_regex = r"score=([\d\.]+), label=happy1"
neutral_regex = r"score=([\d\.]+), label=neutral1"

# Ajout des colonnes pour les scores "happy" et "neutral"
patterns_df = patterns_df.withColumn("happy_score", regexp_extract(col("avatar_emotions_str"), happy_regex, 1).cast("float")) \
                 .withColumn("neutral_score", regexp_extract(col("avatar_emotions_str"), neutral_regex, 1).cast("float"))

# Calcul des moyennes
patterns_df = patterns_df.withColumn("avg_happy_avatar", expr("avg(happy_score)").over(window_spec)) \
                   .withColumn("avg_neutral_avatar", expr("avg(neutral_score)").over(window_spec))
# Fusionner l'array en une seule chaîne
patterns_df = patterns_df.withColumn("avatar_emotions_str", concat_ws(", ", col("avatar_emotions")))'''

display(patterns_df)


# COMMAND ----------

# Define the target path
output_path = "dbfs:/FileStore/patterns_df_ejt"

# Write the DataFrame in Parquet format (you can change to CSV, JSON, etc.)
patterns_df.write.mode("overwrite").parquet(output_path)

# COMMAND ----------

# The path
input_path = "dbfs:/FileStore/patterns_df_ejt"
patterns_df = spark.read.parquet(input_path)

# Show the first few rows
display(patterns_df)

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                    COMPARABLY SCRAPED FROM SCRAPPED COMPANY WEBSITE 
------------------------------------------------------------------------------------------------'''
# File path
file_path = "/FileStore/tables/companies_scraped-1.csv"

# ----------- POSTER EXAMPLE -----------

companies = companies.filter(col('name')=='ICF')

# Read CSV into a PySpark DataFrame
companies_scraped = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(file_path)

# Remove duplicate rows in the DataFrame
companies_scraped = companies_scraped.distinct()

# Perform an inner join to keep only matching rows and add the 'about' column
companies_text_for_values = companies_scraped.join(
    companies, 
    companies_scraped["company_name"] == companies["name"], 
    "inner"
).select(
    companies_scraped["*"],  # Keep all columns from companies_scraped
    companies["about"],
    companies["slogan"]
)

# Display the resulting DataFrame
display(companies_text_for_values)

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                    EXTRACT KEYWORDS FROM COMPANIES DATAFRAME
------------------------------------------------------------------------------------------------'''

# ----------------- FROM 'ABOUT' SECTION -----------------

# Extract KEYWORDS

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import nltk

# Step 1: Define UDF with NLTK Resource Download and Filters
def extract_keywords_nltk(text):
    if text is None:
        return []

    # Ensure NLTK resources are available on executors
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk import pos_tag

    # Tokenize and tag POS
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Extract nouns and adjectives, lowercase, and filter for letters and hyphens
    stop_words = set(stopwords.words('english'))
    keywords = [
        word.lower() for word, pos in pos_tags
        if (pos.startswith('NN') or pos.startswith('JJ'))  # Only nouns and adjectives
        and word.lower() not in stop_words  # Remove stopwords
        and all(c.isalpha() or c == '-' for c in word)  # Allow only letters and hyphens
    ]

    # Return distinct keywords
    return list(set(keywords))

# Step 2: Register UDF
nltk_udf = udf(extract_keywords_nltk, ArrayType(StringType()))


# FROM ABOUT & SLOGAN & ABOUT US SECTIONS
companies_text_for_values = companies_text_for_values.withColumn("about_keywords", nltk_udf(companies_text_for_values["about"]))
companies_text_for_values = companies_text_for_values.withColumn("mvv_keywords", nltk_udf(companies_text_for_values["scraped_data"]))

companies_text_for_values = companies_text_for_values.withColumn("slogan_keywords", nltk_udf(companies_text_for_values["slogan"]))




# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                    EXTRACT VALUES FROM COMPANIES DATAFRAME - MODEL BASED
------------------------------------------------------------------------------------------------'''

'''# ------------------------------- MODEL BASED - WITH CONTEXT -------------------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

# Define the label columns and threshold
THRESHOLD = 0.3
LABEL_COLUMNS = ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
                 'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal',
                 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
                 'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
                 'Universalism: objectivity']

# Define a function to apply the model to a single string
def apply_model_to_text(text):
    if not text:
        return None

    # Load the model and tokenizer inside the function
    tokenizer = AutoTokenizer.from_pretrained("tum-nlp/Deberta_Human_Value_Detector")
    model = AutoModelForSequenceClassification.from_pretrained("tum-nlp/Deberta_Human_Value_Detector", trust_remote_code=True)

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        prediction = model(encoding["input_ids"], encoding["attention_mask"])
        prediction = prediction["output"].flatten().numpy()

    # Filter labels based on threshold
    results = []
    for label, score in zip(LABEL_COLUMNS, prediction):
        if score >= THRESHOLD:
            results.append(f"{label}: {score:.4f}")
    return results


# Define a UDF for the function
@udf(ArrayType(StringType()))
def apply_model_udf(text):
    result = apply_model_to_text(text)
    # Extract the first word before the first colon
    extracted_words = [item.split(':')[0] for item in result]
    return extracted_words


# FROM ABOUT & SLOGAN & ABOUT US SECTIONS
companies = companies.withColumn("about_context", apply_model_udf(companies["about"]))
companies = companies.withColumn("slogan_context", apply_model_udf(companies["slogan"]))
'''


# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                            GATHER VALUES IN COMPANIES DATAFRAME 
------------------------------------------------------------------------------------------------'''

from pyspark.sql.functions import col, concat, array_distinct, coalesce, lit

# Concatenate all six lists into 'keywords_agg' and remove duplicates
companies_text_for_values = companies_text_for_values.withColumn(
    "keywords_agg",
        concat(
            coalesce(col("mvv_keywords"), lit([])),
            coalesce(col("about_keywords"), lit([])),
        )
)



# COMMAND ----------

display(companies_text_for_values)

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                            INFER VALUES IN COMPANIES DATAFRAME
------------------------------------------------------------------------------------------------'''

# ------------------------------- FINAL STEP Infer Companies Core Values -------------------------------

import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_word(word):
    if word:
        return word
    return lemmatizer.lemmatize(word.lower(), pos='n')

from nltk.corpus import wordnet

def filter_and_map_keywords(keywords, core_values):
    matched_values = []
    for kw in keywords:
        kw_lemma = lemmatize_word(kw)
        found_core = None
        for core_val, synset in core_values_lexical_field_with_stemmed_and_lemmatized.items():
            if kw_lemma in synset:
                found_core = core_val
                break
        if found_core:
            matched_values.append(found_core)
    return matched_values

core_values_lexical_field_with_stemmed_and_lemmatized = get_values_dict()
# 1) Broadcast the expanded_core_values dictionary to all workers
bc_core_values_lexical_field_with_stemmed_and_lemmatized = spark.sparkContext.broadcast(core_values_lexical_field_with_stemmed_and_lemmatized)

# 2) Import PySpark functions and types
from pyspark.sql.functions import udf, col, array_union, array_distinct
from pyspark.sql.types import ArrayType, StringType

@udf(returnType=ArrayType(StringType()))
def filter_map_udf(keywords):
    """
    A PySpark UDF that takes a list of keywords (ArrayType(StringType))
    and returns a list of matched core values.
    """
    if keywords is None:
        return []
    
    # Get the dict from broadcast variable
    local_expanded_core_values = bc_core_values_lexical_field_with_stemmed_and_lemmatized.value
    
    # Use your existing function
    matched = filter_and_map_keywords(keywords, local_expanded_core_values)
    return matched

from pyspark.sql.functions import col, array_distinct

# Apply the UDF to extract matched core values and ensure distinct values in the list
companies_text_for_values = (
    companies_text_for_values
    .withColumn("company_values_not_sourced", array_distinct(filter_map_udf(col("keywords_agg"))))
)

companies_values = (
    companies_text_for_values
    .withColumn("company_values_sourced", array_distinct(filter_map_udf(col("slogan_keywords"))))
)

display(companies_values)


# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                    LOGO THEME FROM COMPANIES DATAFRAME
------------------------------------------------------------------------------------------------'''

!pip install pillow scikit-learn
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


import requests
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType

# Define the Python function to get dominant colors
def get_dominant_colors_from_url(url, n_colors=2):
    try:
        # Download the image from the URL
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image = Image.open(response.raw).convert("RGB")
            image = image.resize((100, 100))  # Resize to speed up processing
            image_array = np.array(image).reshape((-1, 3))
            
            # Apply KMeans clustering to find dominant colors
            kmeans = KMeans(n_clusters=n_colors, random_state=0)
            kmeans.fit(image_array)
            dominant_colors = kmeans.cluster_centers_.astype(int).tolist()  # Convert to list for Spark compatibility
            return dominant_colors
        else:
            return None  # Return None if the image cannot be fetched
    except Exception as e:
        return None  # Return None if any error occurs

# Register the function as a UDF
dominant_colors_udf = udf(lambda url: get_dominant_colors_from_url(url), ArrayType(ArrayType(IntegerType())))

# Add the 'dominant_colors_logo' column
companies_with_colors = companies.withColumn("dominant_colors_logo", dominant_colors_udf(companies["logo"]))

# Show the result
display(companies_with_colors)


# COMMAND ----------

# MAGIC %md # MODEL INTERPRETABILITY BASED

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                        IDENTIFYING COMPANY PATTERNS - MODEL INTERPRETABILITY BASED
------------------------------------------------------------------------------------------------'''

# COMMAND ----------

# MAGIC %md # Visualization

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                            VISUALIZE COMPANIES FEATURES DISTRIBUTIONS
------------------------------------------------------------------------------------------------'''

import matplotlib.pyplot as plt
from pyspark.sql.functions import col
import pandas as pd

# Filter out null values in the 'degree' column
filtered_df = patterns_df.filter(col("degree").isNotNull())

# Group by the 'degree' column and count occurrences
degree_counts = filtered_df.groupBy("degree").count().toPandas()

# Define the desired order of degrees
desired_order = ["Associate", "Bachelor", "Master", "Doctorate"]

# Reorder the DataFrame based on the desired order
degree_counts["degree"] = pd.Categorical(degree_counts["degree"], categories=desired_order, ordered=True)
degree_counts = degree_counts.sort_values("degree")

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(degree_counts["degree"], degree_counts["count"], color="skyblue", edgecolor="black")

# Add labels and title
plt.title("Distribution of Degree Types", fontsize=16)
plt.xlabel("Degree Type", fontsize=14)
plt.ylabel("Count", fontsize=14)

# Show the counts on top of each bar
for i, count in enumerate(degree_counts["count"]):
    plt.text(i, count + 5, str(count), ha="center", fontsize=12)

# Display the plot
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert the column 'average_months_of_experience' to a Pandas DataFrame
average_months_df = patterns_df.select("average_months_of_experience").toPandas()

# Drop null values to avoid errors
average_months_df = average_months_df.dropna()

# Extract the column as a Pandas Series
average_months_series = average_months_df["average_months_of_experience"]

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(average_months_series, bins=200, edgecolor='black', alpha=0.7)

# Add titles and labels
plt.title("Distribution of Average Months of Professional Experience", fontsize=16)
plt.xlabel("Average Months of Experience", fontsize=14)
plt.ylabel("Number of Employees", fontsize=14)

# Focus on the range 0 to 600
plt.xlim(0, 300)

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Étape 1 : Compter les occurrences des valeurs dans la colonne 'volunteering'
counts_df = patterns_df.groupBy("top_university").count()

# Étape 2 : Convertir en Pandas DataFrame pour visualisation
counts_pandas = counts_df.toPandas()

# Étape 3 : Tracer le pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    counts_pandas['count'], 
    labels=counts_pandas['top_university'], 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=['lightblue', 'lightgreen']
)
plt.title("Distribution of the 'top_university' variable")
plt.show()

# COMMAND ----------


import matplotlib.pyplot as plt

# Étape 1 : Compter les occurrences des valeurs dans la colonne 'volunteering'
counts_df = patterns_df.groupBy("volunteering").count()

# Étape 2 : Convertir en Pandas DataFrame pour visualisation
counts_pandas = counts_df.toPandas()

# Étape 3 : Tracer le pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    counts_pandas['count'], 
    labels=counts_pandas['volunteering'], 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=['lightblue', 'lightgreen']
)
plt.title("Distribution of the 'volunteering' variable")
plt.show()

# COMMAND ----------


'''VISUALIZATION DISTRIBUTION ALL AVERAGE MONTHS OF EXPERIENCE VS ONE '''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql.functions import col

def plot_avg_experience_density(company_name):
    """
    Plot the density distribution of average months of experience for a specific company
    and for all employees in the dataset.

    Args:
        company_name (str): The name of the company to analyze.
        patterns_df (DataFrame): Spark DataFrame containing 'company_name' and 'average_months_of_experience'.

    Returns:
        None: Displays the plot.
    """
    # Filter data for the specific company
    company_df = patterns_df.filter(col('company_name') == company_name).select("average_months_of_experience")
    company_experience = company_df.toPandas()["average_months_of_experience"].dropna()

    # Get the overall data
    overall_df = patterns_df.select("average_months_of_experience")
    overall_experience = overall_df.toPandas()["average_months_of_experience"].dropna()

    # Ensure there is sufficient data
    if company_experience.empty or overall_experience.empty:
        print(f"Insufficient data for plotting. Check the data for company: {company_name}.")
        return

    # Plot the density distributions
    plt.figure(figsize=(10, 6))
    
    # Plot density for the company
    sns.kdeplot(company_experience, label=f"{company_name} Employees", fill=True, alpha=0.5, color="blue")
    
    # Plot density for the overall data
    sns.kdeplot(overall_experience, label="All Employees", fill=True, alpha=0.5, color="green")

    # Add titles and labels
    plt.title("Density Distribution of Average Months of Experience", fontsize=16)
    plt.xlabel("Average Months of Experience", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(0, 300)
    # Display the plot
    plt.show()


# COMMAND ----------

# MAGIC %md # STATISTIC TESTS

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                                        STATISTIC TESTS
------------------------------------------------------------------------------------------------'''



# COMMAND ----------

'''TEST STAT FOR VOLUNTEERING'''


from statsmodels.stats.proportion import proportions_ztest
from pyspark.sql.functions import col

def test_volunteering_significance(company_name):
    """
    Test if the percentage of volunteering in a specific company is significantly different
    from the overall percentage of volunteering across all employees.

    Args:
        company_name (str): The name of the company to test.
        patterns_df (DataFrame): Spark DataFrame containing 'company_name', 'employee_id', and 'volunteering'.

    Returns:
        dict: Results of the Z-test, including Z-statistic and p-value.
    """
    # Filter data for the specific company
    company_df = patterns_df.filter(col('company_name') == company_name)
    company_volunteers = company_df.filter(col('volunteering') == 1).count()
    company_total = company_df.count()

    # Aggregate overall data
    total_volunteers = patterns_df.filter(col('volunteering') == 1).count()
    total_employees = patterns_df.count()

    # Ensure valid data for the test
    if company_total == 0 or total_employees == 0:
        return {"error": "No data available for the specified company or overall dataset."}

    # Perform the two-proportion Z-test
    count = [company_volunteers, total_volunteers]
    nobs = [company_total, total_employees]
    stat, p_value = proportions_ztest(count, nobs)

    # Interpret the results
    alpha = 0.05
    significance = "significant" if p_value < alpha else "not significant"

    return {
        "company_name": company_name,
        "company_volunteers": company_volunteers,
        "company_total": company_total,
        "overall_volunteers": total_volunteers,
        "overall_total": total_employees,
        "z_statistic": stat,
        "p_value": p_value,
        "significance": significance,
    }


# COMMAND ----------

'''TEST STAT FOR TOP UNIVERSITY'''

from statsmodels.stats.proportion import proportions_ztest
from pyspark.sql.functions import col

def test_top_university_significance(company_name):
    """
    Test if the percentage of employees from a 'top university' in a specific company
    is significantly different from the overall percentage across all employees.

    Args:
        company_name (str): The name of the company to test.
        patterns_df (DataFrame): Spark DataFrame containing 'company_name', 'employee_id', and 'top_university'.

    Returns:
        dict: Results of the Z-test, including Z-statistic and p-value.
    """
    # Filter data for the specific company
    company_df = patterns_df.filter(col('company_name') == company_name)
    company_top_university = company_df.filter(col('top_university') == 1).count()
    company_total = company_df.count()

    # Aggregate overall data
    total_top_university = patterns_df.filter(col('top_university') == 1).count()
    total_employees = patterns_df.count()

    # Ensure valid data for the test
    if company_total == 0 or total_employees == 0:
        return {"error": "No data available for the specified company or overall dataset."}

    # Perform the two-proportion Z-test
    count = [company_top_university, total_top_university]
    nobs = [company_total, total_employees]
    stat, p_value = proportions_ztest(count, nobs)

    # Interpret the results
    alpha = 0.05
    significance = "significant" if p_value < alpha else "not significant"

    return {
        "company_name": company_name,
        "company_top_university": company_top_university,
        "company_total": company_total,
        "overall_top_university": total_top_university,
        "overall_total": total_employees,
        "z_statistic": stat,
        "p_value": p_value,
        "significance": significance,
    }


# COMMAND ----------

'''TEST STAT FOR AVERAGE MONTHS OF EXPERIENCE'''


from scipy.stats import ttest_ind
from pyspark.sql.functions import col

def test_avg_months_experience_significance(company_name):
    """
    Test if the average months of experience in a specific company is significantly different
    from the overall average months of experience across all employees.

    Args:
        company_name (str): The name of the company to test.
        patterns_df (DataFrame): Spark DataFrame containing 'company_name' and 'average_months_of_experience'.

    Returns:
        dict: Results of the t-test, including t-statistic and p-value.
    """
    # Filter data for the specific company
    company_df = patterns_df.filter(col('company_name') == company_name).select("average_months_of_experience")
    company_experience = company_df.toPandas()["average_months_of_experience"].dropna()  # Convert to Pandas Series

    # Aggregate overall data
    overall_df = patterns_df.select("average_months_of_experience")
    overall_experience = overall_df.toPandas()["average_months_of_experience"].dropna()  # Convert to Pandas Series

    # Ensure there is sufficient data
    if len(company_experience) == 0 or len(overall_experience) == 0:
        return {"error": "No data available for the specified company or overall dataset."}

    # Perform the two-sample t-test
    stat, p_value = ttest_ind(company_experience, overall_experience, equal_var=False)  # Welch's t-test

    # Interpret the results
    alpha = 0.05
    significance = "significant" if p_value < alpha else "not significant"

    return {
        "company_name": company_name,
        "company_avg_experience": company_experience.mean(),
        "company_count": len(company_experience),
        "overall_avg_experience": overall_experience.mean(),
        "overall_count": len(overall_experience),
        "t_statistic": stat,
        "p_value": p_value,
        "significance": significance,
    }


# COMMAND ----------

'''TEST STAT FOR Doctorate'''


from statsmodels.stats.proportion import proportions_ztest
from pyspark.sql.functions import col

def test_doctorate_significance(company_name):
    """
    Test if the proportion of employees with a Doctorate degree in a specific company
    is significantly different from the overall proportion across all employees.

    Args:
        company_name (str): The name of the company to test.
        patterns_df (DataFrame): Spark DataFrame containing 'company_name' and 'degree'.

    Returns:
        dict: Results of the Z-test, including Z-statistic and p-value.
    """
    # Filter data for the specific company
    company_df = patterns_df.filter(col('company_name') == company_name)
    company_doctorate = company_df.filter(col('degree') == 'Doctorate').count()
    company_total = company_df.count()

    # Aggregate overall data
    total_doctorate = patterns_df.filter(col('degree') == 'Doctorate').count()
    total_employees = patterns_df.count()

    # Ensure valid data for the test
    if company_total == 0 or total_employees == 0:
        return {"error": "No data available for the specified company or overall dataset."}

    # Perform the two-proportion Z-test
    count = [company_doctorate, total_doctorate]
    nobs = [company_total, total_employees]
    stat, p_value = proportions_ztest(count, nobs)

    # Interpret the results
    alpha = 0.05
    significance = "significant" if p_value < alpha else "not significant"

    return {
        "company_name": company_name,
        "company_doctorate": company_doctorate,
        "company_total": company_total,
        "overall_doctorate": total_doctorate,
        "overall_total": total_employees,
        "z_statistic": stat,
        "p_value": p_value,
        "significance": significance,
    }



# COMMAND ----------

'''TEST STAT FOR MASTER'''


from statsmodels.stats.proportion import proportions_ztest
from pyspark.sql.functions import col

def test_master_significance(company_name):
    """
    Test if the proportion of employees with a Master degree in a specific company
    is significantly different from the overall proportion across all employees.

    Args:
        company_name (str): The name of the company to test.
        patterns_df (DataFrame): Spark DataFrame containing 'company_name' and 'degree'.

    Returns:
        dict: Results of the Z-test, including Z-statistic and p-value.
    """
    # Filter data for the specific company
    company_df = patterns_df.filter(col('company_name') == company_name)
    company_master = company_df.filter(col('degree') == 'Master').count()
    company_total = company_df.count()

    # Aggregate overall data
    total_master = patterns_df.filter(col('degree') == 'Master').count()
    total_employees = patterns_df.count()

    # Ensure valid data for the test
    if company_total == 0 or total_employees == 0:
        return {"error": "No data available for the specified company or overall dataset."}

    # Perform the two-proportion Z-test
    count = [company_master, total_master]
    nobs = [company_total, total_employees]
    stat, p_value = proportions_ztest(count, nobs)

    # Interpret the results
    alpha = 0.05
    significance = "significant" if p_value < alpha else "not significant"

    return {
        "company_name": company_name,
        "company_master": company_master,
        "company_total": company_total,
        "overall_master": total_master,
        "overall_total": total_employees,
        "z_statistic": stat,
        "p_value": p_value,
        "significance": significance,
    }




# COMMAND ----------

# MAGIC %md # ALGORITHM INSTRUCTIONS PROMPT - FINAL STEP

# COMMAND ----------

'''--------------------------------------------------------------------------------------------
                            ALGORITHM INSTRUCTIONS PROMPT - FINAL STEP
------------------------------------------------------------------------------------------------'''

# COMMAND ----------

# TEST

from pyspark.sql.functions import col, lit, when, expr

from pyspark import StorageLevel

'''# Persist the DataFrame in memory
patterns_df.cache()

# Trigger materialization
patterns_df.count()'''

def enhance_profile(profile_id):
    # Filtrer les données pour le profil donné
    filtered_df = patterns_df.filter(col('employee_id') == profile_id)
    
    # Initialiser un dictionnaire pour stocker les éléments du profil
    profile_elements = {'employee_id': profile_id}
    
    if filtered_df.count() > 0:  # Vérifier si le profil existe
        row = filtered_df.first()
        
        # Ajouter les éléments standard
        profile_elements['degree'] = row['degrees']
        profile_elements['avg_post_duration'] = row['average_months_of_experience']

        # Ajouter les détails sur l'université si top_university == 1
        if row['top_university'] == 1:
            # Ajouter directement le détail de l'université
            profile_elements['top_university'] = row['educations_details']
            
            # Extraire les valeurs de 'values_from_feature.top_university'
            values_from_feature = row['values_from_features']  # C'est un objet/dict
            education_details = row['educations_details']  # La valeur à associer
            
            # Vérifier si 'top_university' existe dans 'values_from_feature'
            if 'top_university' in values_from_feature and values_from_feature['top_university']:
                # Créer un dictionnaire basé sur les valeurs de 'top_university'
                values_sourced = {
                    value: ('top_university', education_details) for value in values_from_feature['top_university']
                }
                
                # Ajouter au dictionnaire final
                profile_elements['values_sourced'] = values_sourced
            else:
                # Si 'top_university' n'existe pas ou est null
                profile_elements['values_sourced'] = {}
        
        # Ajouter les causes de volontariat si volunteering == 1
        if row['volunteering'] == 1:
            volunteer_experience_df = filtered_df.withColumn(
                'experience', explode(col('volunteer_experience'))
            ).select(
                when(col('experience.cause').isNotNull(), col('experience.cause')).otherwise(lit(0)).alias('cause')
            )
            
            causes = volunteer_experience_df.select(collect_set('cause').alias('causes')).first()['causes']
            profile_elements['volunteer'] = causes
        
        # Ajouter les 'values_notsourced' depuis 'values_from_feature'
        values_from_feature = row['values_from_features']
        if values_from_feature and 'degrees' in values_from_feature and values_from_feature['degrees']:
            profile_elements['values_notsourced'] = values_from_feature['degrees']
        else:
            profile_elements['values_notsourced'] = []
        values_from_feature = row['values_from_features']
        
        if values_from_feature and 'company_and_organization_types' in values_from_feature and values_from_feature['company_and_organization_types']:
            profile_elements['values_notsourced'] = values_from_feature['company_and_organization_types']
        
        
        profile_elements['values_notsourced'] = row['profile_values_not_sourced']
        values_from_volunteering = row['values_from_volunteering']

        # Vérifier si la colonne n'est pas nulle
        if values_from_volunteering:
            # Convertir le contenu en une structure lisible ou le manipuler
            profile_elements['values_sourced'] = {
                key: value for key, value in values_from_volunteering.items()
            }
    return profile_elements



# COMMAND ----------

id='abad-shah-32942b36'
result3= enhance_profile(id)
print(result3)

# COMMAND ----------

display(patterns_df.filter(col('id') == 'abad-shah-32942b36'))

# COMMAND ----------


# Exemple d'utilisation
profile_id2 = 'aariel-allen-155790118'
profile_id1 = 'adam-guilmette-68940236'
profile_id ='abhinav-garg-7841ab33'
id2='aariel-allen-155790118'
result = enhance_profile(profile_id)
result1 = enhance_profile(profile_id1)
result2= enhance_profile(profile_id2)
res=enhance_profile(id2)
print(res)
print(result)
print(result1)
print(result2)

# COMMAND ----------

display(patterns_df)

# COMMAND ----------

from pyspark.sql.functions import col

def enhance_company(company_name):
    """
    Enhance the profile of a company by testing the significance of various features.

    Args:
        company_name (str): The name of the company to analyze.

    Returns:
        dict: A dictionary with significant features sorted by p-value in descending order.
    """

    # Filter the data for the specific company
    filtered_df = patterns_df.filter(col('company_name') == company_name)

    # Initialize a dictionary to store company profile elements
    company_elements = {'company_name': company_name}

    # Ensure there is data for the given company
    if filtered_df.count() > 0:

        # Perform all the tests
        volunteering_test = test_volunteering_significance(company_name)
        top_university_test = test_top_university_significance(company_name)
        avg_months_test = test_avg_months_experience_significance(company_name)
        doctorate_test = test_doctorate_significance(company_name)
        master_test = test_master_significance(company_name)

        # List of tests to evaluate
        tests = {
            "volunteering": volunteering_test,
            "top_university": top_university_test,
            "average_months_of_experience": avg_months_test,
            "doctorate": doctorate_test,
            "master": master_test,
        }

        # Add significant features and p-values to the dictionary
        significant_features = {
            feature: result["p_value"]
            for feature, result in tests.items()
            if "p_value" in result and result["p_value"] < 0.05
        }

        # Sort the significant features by p-value in descending order
        sorted_features = dict(sorted(significant_features.items(), key=lambda x: x[1], reverse=True))

        # Add sorted significant features to company_elements
        company_elements.update(sorted_features)

    else:
        company_elements["error"] = "No data available for the specified company."

    return company_elements


# COMMAND ----------

# ----------------------------------------------------------------------------------------------------------------------------------------------------
#                                                           THE FINAL FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def enhance_profile(profile_id, company_name):

    profile_elements = {}
    company_elements = {}

    # IMPLEMENT HERE

    return profile_elements, company_elements

def matching(profile_elements, company_elements, profile_id, company_name):
    matching = {}

    for feature in prof

    return matching


def get_matching(profile_id, company_name):
    profile_elements, company_elements = enhance_profile(profile_id, company_name)
    return matching(profile_elements, company_elements, profile_id, company_name)

# COMMAND ----------

!huggingface-cli login --token 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
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base")

# COMMAND ----------

# Define a simple text prompt
prompt = "Imagine a person named Joey who has volunteered. Most employees of the company [GreenTech] did too. Say to Joey to highlight that he has volunteered in his resume"

# Generate the text with adjusted parameters
output = pipe(
    prompt,
    max_length=400,          # Total length of the text including the prompt
    max_new_tokens=500,      # Tokens to generate beyond the prompt
    temperature=0.7,         # Controls randomness (0.7 for balanced creativity)
    top_p=0.9,               # Nucleus sampling for coherent yet varied results
    top_k=50,                # Limits the token pool to top 50 at each step
    do_sample=True,          # Enables probabilistic sampling for diversity
    repetition_penalty=1.2,  # Penalizes repetitive tokens
    length_penalty=0.8,      # Slightly favors longer text
    num_return_sequences=1,  # Generates a single response
    early_stopping=True      # Stops on an end-of-sequence token
)

# Print the generated response
print(output[0]['generated_text'])


# COMMAND ----------

# ---------------------------------------------------------------------------------------------------------------------------------
#                                                     BOOKS
#--------------------------------------------------------------------------------------------------------------------------------

def get_books():
    books = {}

    
    return books

# COMMAND ----------

# ---------------------------------------------------------------------------------------------------------------------------------
#                                                     THE INSTRUCTION PDF FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------

def get_matching(profile_elements, company_elements, profile_id, company_name):

    # ------------------- VALUES -------------------
    # Initialize matched_elements with 'values' as an empty list.
    matched_elements = {'values': []}
    
    # Check if the company has a non-empty 'values_slogan' list.
    if 'values_slogan' in company_elements and company_elements['values_slogan']:
        for slogan in company_elements['values_slogan']:
            # If the slogan is in the profile's 'values_not_sourced' list, append a tuple.
            if slogan in profile_elements.get('values_not_sourced', []):
                matched_elements['values'].append((slogan, None, 'slogan'))
            
            # If the slogan is a key in profile_elements['values_sourced'],
            # retrieve the first element of the first inner list and append a tuple.
            if slogan in profile_elements.get('values_sourced', {}):
                sourced_value = profile_elements['values_sourced'][slogan]
                # Verify that sourced_value is a non-empty list and its first element is also non-empty.
                if sourced_value and isinstance(sourced_value, list) and len(sourced_value) > 0 and sourced_value[0]:
                    first_inner_list = sourced_value[0]
                    if isinstance(first_inner_list, list) and len(first_inner_list) > 0:
                        first_element = first_inner_list[0]
                        matched_elements['values'].append((slogan, first_element, 'slogan'))
      
    # Process values_not_sourced from company_elements.
    if 'values_not_sourced' in company_elements and company_elements['values_not_sourced']:
        for value in company_elements['values_not_sourced']:
            # If the value is in profile's values_not_sourced, append (value, None, None).
            if value in profile_elements.get('values_not_sourced', []):
                matched_elements['values'].append((value, None, None))
            
            # If the value is a key in profile's values_sourced, append (value, first_element, None).
            if value in profile_elements.get('values_sourced', {}):
                sourced_value = profile_elements['values_sourced'][value]
                if (sourced_value and isinstance(sourced_value, list) and len(sourced_value) > 0 
                        and sourced_value[0] and isinstance(sourced_value[0], list) and len(sourced_value[0]) > 0):
                    first_element = sourced_value[0][0]
                    matched_elements['values'].append((value, first_element, None))

                  
      # Deduplicate tuples based on the first element.
    # For each key, keep the tuple with fewer None values in its second and third positions.
    best = {}
    for tpl in matched_elements['values']:
        key = tpl[0]
        # Count the number of None values in positions 1 and 2.
        none_count = (1 if tpl[1] is None else 0) + (1 if tpl[2] is None else 0)
        if key not in best:
            best[key] = tpl
        else:
            existing = best[key]
            existing_none_count = (1 if existing[1] is None else 0) + (1 if existing[2] is None else 0)
            # If the current tuple has fewer None values, replace the existing tuple.
            if none_count < existing_none_count:
                best[key] = tpl
    
    # Replace the values list with the deduplicated tuples.
    matched_elements['values'] = list(best.values())

    # ------------------- PATTERNS -------------------
    
    # Initialize matched_elements with 'patterns' as an empty list.
    matched_elements['patterns'] = []

    # Rank specific numerical columns in ascending order.
    rank_columns = ['master', 'doctorate', 'volunteer', 'average_tenure', 'top_university']
    filtered_columns = {col: company_elements[col] for col in rank_columns if col in company_elements and isinstance(company_elements[col], (int, float))}
    
    # Sort column names by their values in ascending order.
    company_patterns = sorted(filtered_columns, key=filtered_columns.get)

    # Process each pattern in the sorted list.
    for pattern in company_patterns:
        # For the volunteering case:
        # Note: Company uses the key "volunteering" (in rank_columns), but profile_elements holds "volunteer".
        if pattern == 'volunteer' and 'volunteer' in profile_elements and len(profile_elements['volunteer'])>0 :
            matched_elements['patterns'].append('volunteer')
        
        # For the top_university case:
        if pattern == 'top_university' and 'top_university' in profile_elements and profile_elements['top_university']:
            # Append a string with the value from profile_elements inserted.
            matched_elements['patterns'].append(
                "top_university:{}".format(profile_elements['top_university'])
            )

        # For the master case:
        # If in profile_elements['degree'] there is 'master', append 'master'
        # but only if 'doctorate' is not already in the patterns.
        if pattern == 'master':
            if 'degree' in profile_elements and 'Master' in profile_elements['degree']:
                if 'doctorate' not in matched_elements['patterns']:
                    matched_elements['patterns'].append('master')
        
        # For the doctorate case:
        # If in profile_elements['degree'] there is 'doctorate', append 'doctorate'
        # and remove 'master' from patterns if it exists.
        if pattern == 'doctorate':
            if 'degree' in profile_elements and 'Doctorate' in profile_elements['degree']:
                matched_elements['patterns'].append('doctorate')
                if 'master' in matched_elements['patterns']:
                    matched_elements['patterns'].remove('master')
        
        # ADD AVERAGE TENURE 

    # ------------------- SYNERGIES -------------------

    if 'picture' in company_elements and 'picture' in profile_elements:
        if company_elements['picture'] == profile_elements['picture']:
            matched_elements['picture'] = [company_elements['picture'], 1]
        else:
            matched_elements['picture'] = [company_elements['picture'], 0]

    matched_elements['theme'] = company_elements['logo_theme']


    return matched_elements



def generate_instructions(profile_id, company_name, profile_elements, company_elements):

    matched_elements = get_matching(profile_elements, company_elements, profile_id, company_name)

    books = get_books()

    prompts = []

    return prompts



def generate_pdf(prompts):

    return generated



# FINAL USER FUNCTION FOR FRONT END INTERFACE
def user_function(company_id, profile_id):
    prompts = generate_instructions(profile_id, company_id)
    generate_pdf(prompts)


# COMMAND ----------

# MAGIC %md # --- TESTING AREA ---

# COMMAND ----------

from pyspark.sql.functions import col, size

# Assuming your DataFrame is called 'df'
profiles_filtered = profiles.filter((col("about").isNotNull()) & (size(col("recommendations")) > 1) & (size(col("experience")) > 2) & (size(col("volunteer_experience")) > 1) & (col('top_university') == lit(1)))

# Display the filtered DataFrame
display(profiles_filtered)


# COMMAND ----------

from pyspark.sql.functions import col, size

# Assuming your DataFrame is called 'df'
companies_filtered = companies_values.filter((col("about").isNotNull()) & (col("slogan").isNotNull()) & (col("scraped_data").isNotNull()))

# Display the filtered DataFrame
display(companies_filtered)

# COMMAND ----------

profile_example_id = 'carayou'

# COMMAND ----------

company_example_name = 'ICF'

# COMMAND ----------

display(companies.filter(col('name') == company_example_name))
