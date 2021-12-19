# The dataset is pulled from the Singapore's monitoring cameras using their data.gov.sg API calls
# The most appropriate pictures are from the MCE and one from between MCE and ECP district cameras.
# Singapore is with 8 hours in before of Romania
# The peak traffic hour period is: 07-09 am
# The least traffic period is:
# The cameras have the ids: 1502, 1504, 1505, 4701, 4706, 4708, 4714, 4709, 4707, 3704, 3705, 3702,
# 6711, 7793, 7797, 7798
# The API provides the pictures taken from current time stamps or previous ones
# The date_time parameter has the format: YYYY-MM-DD[T]HH:mm:ss (SGT) (ex. 2021-12-17T16:25:49)
# The API Request example is: https://api.data.gov.sg/v1/transport/traffic-images?date_time=2021-12-17T16%3A25%3A49
# The API Response is a JSON
# (ex.
# {
#   "items": [
#     {
#       "timestamp": "2021-12-17T16:25:33+08:00",
#       "cameras": [
#         {
#           "timestamp": "2021-12-17T16:25:33+08:00",
#           "image": "https://images.data.gov.sg/api/traffic-images/2021/12/b54f619f-d90f-45b5-8d11-71ee2e577150.jpg",
#           "location": {
#             "latitude": 1.29531332,
#             "longitude": 103.871146
#           },
#           "camera_id": "1001",
#           "image_metadata": {
#             "height": 240,
#             "width": 320,
#             "md5": "6a305704c633fe455f32c1d70e228835"
#           }
#         },
# )
import html

import requests
import logging
from tqdm import tqdm as progress_bar


class DataGatherer:
    def __init__(self, output_location):
        self.__output_location = output_location
        self.__api_body = "https://api.data.gov.sg/v1/transport/traffic-images"
        self.__camera_ids = [1502, 1504, 1505, 4701, 4706, 4708, 4714, 4709, 4707, 3704, 3705, 3702, 6711, 7793,
                             7797, 7798]
        logging.basicConfig(format="%(asctime)s - [DATA GATHERER] -> %(message)s", datefmt='%d-%b-%y %H:%M:%S',
                            level=logging.INFO)

    def get_traffic_images_feed(self, date_time=""):
        try:
            if date_time != "":
                request = self.__api_body + "?date_time=" + html.escape(date_time)
                response = requests.get(request)
            else:
                request = self.__api_body
                response = requests.get(request)

            if response.status_code == 200:
                logging.info("API call succeeded: Code " + str(response.status_code))

                response = response.json()["items"][0]
                cameras = response["cameras"]

                logging.info("Data saving...")
                for camera_index in progress_bar(range(len(cameras))):
                    # if int(cameras[camera_index]["camera_id"]) in self.__camera_ids:
                    if int(cameras[camera_index]["camera_id"]):
                        image_response = requests.get(cameras[camera_index]["image"])

                        if image_response.status_code == 200:
                            timestamp = cameras[camera_index]["timestamp"].split("T")
                            hour = timestamp[1].split(":")
                            hour = hour[0] + hour[1] + hour[2].split(".")[0]
                            timestamp = timestamp[0] + "-" + hour
                            image_format = timestamp + cameras[camera_index]["camera_id"] + "_" + \
                                           str(camera_index) + ".png"
                            image = open(self.__output_location + image_format, "wb")
                            image.write(image_response.content)
                            image.close()
                        else:
                            logging.critical("Could not retrieve image: Code" + str(image_response.status_code))
                logging.info("Data saved")
            else:
                logging.critical("API call failed: Code " + str(response.status_code) + "-> Request was: " + request)
        except Exception as e:
            logging.critical("Exception occurred while requesting data: " + str(e))
