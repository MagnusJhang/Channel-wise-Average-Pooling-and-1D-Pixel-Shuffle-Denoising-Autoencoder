import json
# from. import ParmModel

class ParmModel:
    # def __init__(self, nlen=None, ks=3, nlayer=None, nch_1=None, model_name=None, *args, **kwargs):
    def __init__(self, nlen=None, model_name=None, *args, **kwargs):
        self.nlen = nlen
        # self.ks = ks
        # self.nlayer = nlayer # Number of encoder-decoder layer
        # self.nch_1 = nch_1
        self.model_name = model_name
        for k, v in kwargs.items():
            if isinstance(v, str):
                exec("self.{} = '{}'".format(k, v))
            else:
                exec("self.{} = {}".format(k, v))
            # try:
            #     exec("self.{} = {}".format(k, v))
            # except NameError:
            #     exec("self.{} = '{}'".format(k, v))


    def to_json_file(self, file):
        json_str = json.dumps(self.__dict__)
        with open(file, 'w') as f:
            f.write(json_str)

    def from_json_file(self, file):
        with open(file, 'r') as f:
            d = json.load(f)
            for k, v in d.items():
                if isinstance(v, str):
                    exec("self.{} = '{}'".format(k, v))
                else:
                    exec("self.{} = {}".format(k, v))

    def add_parm(self, k, v):
        try:
            exec("self.{} = {}".format(k, v))
        except NameError:
            exec("self.{} = '{}'".format(k, v))

    def __str__(self):
        return json.dumps(self.__dict__)