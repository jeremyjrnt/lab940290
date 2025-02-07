# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, count
from pyspark.sql.types import IntegerType, ArrayType, StringType
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.getOrCreate()

companies = spark.read.parquet('/dbfs/linkedin_train_data')
companies.display()

profiles = spark.read.parquet('/dbfs/linkedin_people_train_data')
profiles.display()

input_path = "dbfs:/FileStore/patterns_df_ejt"
patterns_df = spark.read.parquet(input_path)
patterns_df = patterns_df.orderBy(col('company_name').asc(), col('employee_id').asc())

patterns_df.display()

"""Creation of the positive samples"""

train_dataset_positive_labels_df = patterns_df.select('company_name', 'employee_id', 'top_university', 'degree', 'volunteering', 'average_months_of_experience', 'company_and_organization_types')
train_dataset_positive_labels_df = train_dataset_positive_labels_df.withColumn('label', lit(1))
train_dataset_positive_labels_df = train_dataset_positive_labels_df.orderBy(col('company_name').asc(), col('employee_id').asc())

train_dataset_positive_labels_df.display()

"""Creation of the negative samples"""

employees_id = patterns_df.select('employee_id').distinct()
companies_name = patterns_df.select('company_name').distinct()
train_dataset_positive_labels_employee_company_df = train_dataset_positive_labels_df.select('employee_id', 'company_name')

train_dataset_negative_labels_df = employees_id.crossJoin(companies_name)
train_dataset_negative_labels_df = train_dataset_negative_labels_df.subtract(train_dataset_positive_labels_employee_company_df).orderBy(col('employee_id').asc(), col('company_name').asc())
train_dataset_negative_labels_df = train_dataset_negative_labels_df.withColumn('label', lit(0)).withColumnRenamed('company_name', 'negative_company_name')

train_dataset_negative_labels_df = train_dataset_negative_labels_df.join(patterns_df, on=['employee_id'], how='left').select('negative_company_name', 'employee_id', 'top_university', 'degree', 'volunteering', 'average_months_of_experience', 'company_and_organization_types', 'label').orderBy(col('negative_company_name').asc(), col('employee_id').asc())
train_positive_sample_size = train_dataset_positive_labels_df.groupBy("company_name").agg(count("*").alias("sample_size")).select('company_name', 'sample_size')

train_negative_labels_with_company_name = train_dataset_negative_labels_df.crossJoin(train_positive_sample_size)
train_negative_labels_with_company_name = train_negative_labels_with_company_name.filter(col('negative_company_name') == col('company_name'))

train_negative_with_random = train_negative_labels_with_company_name.withColumn("random_val", F.rand())
window = Window.partitionBy("company_name").orderBy("random_val")
train_negative_with_row_num = train_negative_with_random.withColumn("row_num", F.row_number().over(window))
train_negative_samples_by_company = train_negative_with_row_num.filter(F.col("row_num") <= F.col("sample_size"))
train_negative_samples_by_company = train_negative_samples_by_company.drop("row_num", "random_val")

train_negative_samples_by_company.display()

train_negative_samples_by_company.drop('negative_company_name', 'sample_size')
train_negative_samples_by_company = train_negative_samples_by_company.select(train_dataset_positive_labels_df.columns)
train_df = train_dataset_positive_labels_df.union(train_negative_samples_by_company).orderBy(col('company_name').asc(), col('employee_id').asc())

train_df.display()


"""Preprocessing of the features of the employees"""

def degree2feature(degree):
    if degree is None:
        return 0
    
    elif degree == 'Bachelor':
        return 1
    
    elif degree == 'Master':
        return 2
    
    elif degree == 'Doctorate':
        return 3
    
    elif degree == 'Associate':
        return 4
    
def organization_types2string(organization_types):
    organization_types_string = ''
    for dico in organization_types:
        try:
            org_type = dico['organization_type']
            organization_types_string += org_type
            organization_types_string += ' '

        except:
            pass   

    return organization_types_string    

degree2features_udf = udf(degree2feature, IntegerType())
organization_types_2string_udf = udf(organization_types2string, StringType())

train_df = train_df.withColumn('degree_feature', degree2features_udf(col('degree')))
train_df = train_df.withColumn('organization_types', organization_types_2string_udf(col('company_and_organization_types')))

indexer = StringIndexer(inputCol="organization_types", outputCol="organization_types_feature")
train_df = indexer.fit(train_df).transform(train_df)
train_df = train_df.drop('organization_types', 'employee_id', 'degree', 'company_and_organization_types')

train_df.display()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import collect_list, struct
from pyspark.sql.types import ArrayType, StringType

"""Training the models and extraction of the feature importances"""

feature_cols = ['top_university', 'volunteering', 'average_months_of_experience', 'degree_feature', 'organization_types_feature']
train_df = train_df.fillna(-1, subset=['average_months_of_experience'])

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(train_df)

companies_df = train_df.groupBy("company_name").agg(collect_list(struct(*train_df.columns)).alias("employees"))

def train_rf_per_company(company_name, employees_data):
    df_company = spark.createDataFrame(employees_data)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
    model = rf.fit(df_company)

    feature_importances = model.featureImportances.toArray()
    feature_importance_dict = dict(zip(feature_cols, feature_importances))
    sorted_features = sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)

    return (company_name, sorted_features)

results = []

for row in companies_df.collect():
    company_name = row['company_name']
    employees_data = row['employees']
    
    result = train_rf_per_company(company_name, employees_data)
    results.append(result)

result_df = spark.createDataFrame(results, ["company_name", "features_importance"])
result_df.display()
