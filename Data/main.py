from Data.DataGatherer import DataGatherer

if __name__ == "__main__":
    data_gatherer = DataGatherer("Data/output/peakhour/")
    data_gatherer.get_traffic_images_feed("2021-12-08T07:50:00")
