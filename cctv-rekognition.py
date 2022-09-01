#!/usr/bin/env python
# coding: utf-8

# In[68]:


import boto3
import json
import logging
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw
import cv2
import numpy as np
import math
import os


logger = logging.getLogger(__name__)


# In[69]:


def name_selector(raw_name):
    sections = raw_name.split("_")

    channel = sections[1]
    date = sections[3]
    time = sections[4].replace("-",":").split(".")[0]
    date_time = date+" "+time
    
    return channel, date_time



# In[70]:


def channel_selector(channel):

    channel_dict = {
        "ch1": "support",
        "ch2": "developers",
        "ch3": "kitchen",
    }
    
    return channel_dict[channel]
    


# In[71]:


def upload_json_s3(json_fileName, json_data, s3_bucketName):  
    s3 = boto3.resource('s3')
    s3object = s3.Object(s3_bucketName, json_fileName)

    s3object.put(
        Body=(bytes(json_data.encode('UTF-8')))
        )
    


# In[72]:


def face_details_selector(faceDetail, index, date_time, baseName):

    id = baseName + "_" + str(index)
    faceObject = {
        "id": id,
        "date_time": date_time,
    }
    
    for detailName, detail in faceDetail.items():

        if detailName == "Smile":
            faceObject[detailName] = detail['Value']
        elif detailName == "Gender":
            faceObject[detailName] = detail['Value']
        elif detailName == "Emotions":
            for emotion in detail:
                faceObject[emotion['Type']] = emotion['Confidence']
    
    return faceObject

    


# In[73]:


def detect_labels(photo, bucket, json_fileName):

    rekognition=boto3.client('rekognition')
    s3=boto3.client('s3')
    
    try:
        response = rekognition.detect_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}})
        channel, date_time = name_selector(photo)
        detected_persons = {
            "persons_detected": 0,
            "date_time": date_time,
        }
        
        labels = response["Labels"]
        
        for label in labels:
            if label["Name"] == "Person" or label["Name"] == "Human":
                number_of_people = len(label["Instances"])
                detected_persons["persons_detected"] = detected_persons["persons_detected"] + number_of_people
                
    except ClientError:
        logger.info("Couldn't detect labels in %s.", photo)
        raise
    else:
        
        jsonObject = json.dumps(detected_persons)
        print(jsonObject)
 
        upload_json_s3(json_fileName = json_fileName, json_data = jsonObject, s3_bucketName = bucket)
    


# In[74]:


def crop_image(imgArray, boundingBox):
    
    image_height, image_width, c = imgArray.shape
    left = math.floor(image_width * abs(boundingBox['Left']))
    top = math.floor(image_height * abs(boundingBox['Top']))

    height = math.floor(top + image_height * abs(boundingBox['Height']))
    width = math.floor(left + image_width * abs(boundingBox['Width']))

    cropped_arr = imgArray[top:height, left:width]
    return cropped_arr


# In[75]:


def match_face(bucket_name, file_name):
    # file should be in S3 in the mentioned bucket
    bucket=bucket_name
    collectionId='office-collection'
    fileName=file_name
    threshold=70
    maxFaces=2

    client=boto3.client('rekognition')
    try: 
        
        response=client.search_faces_by_image(CollectionId=collectionId,
                                    Image={'S3Object':{'Bucket':bucket,'Name':fileName}},
                                    FaceMatchThreshold=threshold,
                                    MaxFaces=maxFaces)

        if len(response['FaceMatches']) > 0:
            return response['FaceMatches'][0]['Face']['ExternalImageId']
        else:
            return {'ExternalImageId': 'unknown'}
    except ClientError:
        return {'ExternalImageId': 'unknown'}
        raise


# In[76]:


def detect_faces(photo, bucket, baseName, json_directory):
    """
    Detects faces in the image.

    :return: The list of faces found in the image.
    """
    
    
    rekognition=boto3.client('rekognition')
    s3=boto3.client('s3')
    s3_resource = boto3.resource('s3')
    bucket_resource = s3_resource.Bucket(bucket)

    try:
        
        response = rekognition.detect_faces(Image={'S3Object': {'Bucket': bucket, 'Name': photo}}, Attributes=['ALL'])
        
        channel, date_time = name_selector(photo)

        img = bucket_resource.Object(photo).get().get('Body').read()

        imgArray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)        
        
        faceDetails = response['FaceDetails']
        
        for index, faceDetail in enumerate(faceDetails):
            
            selected_details = face_details_selector(faceDetail, index, date_time, baseName)
            
            boundingBox = faceDetail['BoundingBox']

            cropped_image_array = crop_image(imgArray, boundingBox)     

            cropped_image_array = cv2.cvtColor(cropped_image_array, cv2.COLOR_BGR2RGB)

            img_cropped = Image.fromarray(cropped_image_array, 'RGB')
            
            img_cropped_file_name = baseName + str(index) + '.png'
            
            img_cropped_temp_file = 'temp/' + img_cropped_file_name
            
            img_cropped.save(img_cropped_temp_file)
            
            cropped_face_temp_path = 'temp/cropped-face/'+ img_cropped_file_name
            
            s3.upload_file(img_cropped_temp_file, bucket, cropped_face_temp_path)
            
            selected_details['employee_name'] = match_face(bucket, cropped_face_temp_path)
            
            os.remove(img_cropped_temp_file)
            s3_resource.Object(bucket, cropped_face_temp_path).delete()
            
            json_file_name = json_directory + selected_details['id'] + '.json'
            
            jsonObject = json.dumps(selected_details)
            print(jsonObject)
            
            upload_json_s3(json_fileName = json_file_name, json_data = jsonObject, s3_bucketName = bucket)


                
    except ClientError:
        logger.info("Couldn't detect labels in %s.", photo)
        raise


# In[78]:




def main():

    image_s3Bucket = 'cctv-analytics-z1'
    client = boto3.client('s3')
    s3 = boto3.resource('s3')
    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=image_s3Bucket, Prefix='Output-Frames/ch2/')

    for page in page_iterator:
        if page['KeyCount'] > 0:
            for item in page['Contents']:

                image_fileName = item['Key']
                image_fileName_splitted = image_fileName.split('/')
                image_base_name = image_fileName_splitted[3].split('.')[0]
                print(image_base_name)


                channel = channel_selector(image_fileName_splitted[1])
                date = image_fileName_splitted[2]

    #             jsonBaseName = image_base_name + ".json"
    #             json_fileName = f"Output-Json/LabelsDetected/channel={channel}/date={date}/{jsonBaseName}"
    #             detect_labels(image_fileName, image_s3Bucket, json_fileName)

                json_directory_faces = f"Output-Json/FacesDetected/channel={channel}/date={date}/"
                detect_faces(image_fileName, image_s3Bucket, image_base_name, json_directory_faces)


if __name__ == "__main__":
    main()

    
