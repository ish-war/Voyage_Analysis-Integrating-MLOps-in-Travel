class DataTransformer:  
    def __init__(self, data):  
        self.data = data  

    def transform(self):  
        # Dummy transformation (you can modify as needed)
        return self.data.dropna()
