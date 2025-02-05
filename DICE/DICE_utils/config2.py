class bank:
    """
    A new class that repl8icates the data from the original './config.py' bank
    class but uses '@property' for getters and setters.
    """

    def __init__(self):
        # The size of total features
        self._params = -1
        
        # The valid region (bounds of each feature)
        self._input_bounds = []

        # The name of each feature
        self._feature_name = []

        # The name of each class
        # TODO: Specify what this is for? Not in DICE_Search
        self._class_name = ["no", "yes"]

        # Specify the categorical features with their indices
        # TODO: Ask where this is used? Not in DICE_Search, If not needed in Search this info is in categorical_unique_values
        self._categorical_features = []

        # System Message for the LLM
        self._system_message = ''

        # List of encoders that were used on this dataset
        self._label_encoders = {}

        # List of the unique values each categorical feature can have
        self._categorical_unique_values = {}

    #------params------
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        self._params = value

    #------input_bounds------
    @property
    def input_bounds(self):
        return self._input_bounds
    
    @input_bounds.setter
    def input_bounds(self, new_bounds):
        self._input_bounds = new_bounds

    #------feature_name------
    @property
    def feature_name(self):
        return self._feature_name
    
    @feature_name.setter
    def feature_name(self, new_feature_name):
        self._feature_name = new_feature_name
    
    #------class_name------
    @property
    def class_name(self):
        return self._class_name
    
    @class_name.setter
    def class_name(self, new_class_name):
        self._class_name = new_class_name

    #------categorical_features------
    @property
    def categorical_features(self):
        return self._categorical_features
    
    @categorical_features.setter
    def categorical_features(self, new_categorical_features):
        self._categorical_features = new_categorical_features

    #------system_message------
    @property
    def system_message(self):
        return self._system_message
    
    @system_message.setter
    def system_message(self, new_system_message):
        self._system_message = new_system_message

    #------label_encoders------
    @property
    def label_encoders(self):
        return self._label_encoders
    
    @label_encoders.setter
    def label_encoders(self, new_label_encoders):
        self._label_encoders = new_label_encoders

    #------categorical_unique_values------
    @property
    def categorical_unique_values(self):
        return self._categorical_unique_values
    
    @categorical_unique_values.setter
    def categorical_unique_values(self, new_categorical_unique_values):
        self._categorical_unique_values = new_categorical_unique_values
