from mercury import Text
import os
import bcrypt
import json
import glob
import ipywidgets
from IPython.display import display
from traitlets import Unicode
from mercury import WidgetsManager

"""
This moddule creates a Password subclass based on mercury's Text class
that encrypts the password a la ipywidgets.Password.
"""


class customPassword:


    #unauthorized = "This device is not authorized"
    #key = ""
    #incorrect = "Incorrect password, try again"
    def __init__(
        self, value="", label="", rows=1, url_key="", disabled=False, hidden=False, sanitize=True, unauthorized="This device is not authorized",key="",incorrect="Incorrect password, try again"
    ):
        #super().__init__("",label,rows,url_key,disabled,hidden,sanitize)

        #no input value is given, always initialized empty
        self.rows = rows
        self.unauthorized = unauthorized
        self.incorrect = incorrect
        self.key = key
        

        self.code_uid = WidgetsManager.get_code_uid("Text", key=url_key)
        self.url_key = url_key
        self.hidden = hidden
        self.sanitize = sanitize
        if WidgetsManager.widget_exists(self.code_uid):
            self.text = WidgetsManager.get_widget(self.code_uid)
            self.text.description = label
            self.text.disabled = disabled
        else:
            self.text = ipywidgets.Password(
                value=WidgetsManager.get_preset_value(url_key, value), description=label, disabled=disabled
            )
            #self.realvalue = value
            #self.text._view_name = Unicode('PasswordView').tag(sync=True)
            #self.text._model_name = Unicode('PasswordModel').tag(sync=True)
            WidgetsManager.add_widget(self.text.model_id, self.code_uid, self.text)
        display(self)


    @property
    def value(self):
        return self.text.value

    # @value.setter
    # def value(self, v):
    #    self.text.value = v

    def __str__(self):
        return "mercury.Text"

    def __repr__(self):
        return "mercury.Text"

    def _repr_mimebundle_(self, **kwargs):
        # data = {}
        # data["text/plain"] = repr(self)
        # return data
        data = self.text._repr_mimebundle_()

        if len(data) > 1:
            view = {
                "widget": "Text",
                "value": self.text.value,
                "sanitize": self.sanitize,
                "rows": self.rows,
                "label": self.text.description,
                "model_id": self.text.model_id,
                "code_uid": self.code_uid,
                "url_key": self.url_key,
                "disabled": self.text.disabled,
                "hidden": self.hidden,
            }
            data["application/mercury+json"] = json.dumps(view, indent=4)
            if "text/plain" in data:
                del data["text/plain"]

            if self.hidden:
                key = "application/vnd.jupyter.widget-view+json"
                if key in data:
                    del data[key]

            return data


    def checkpw(self):
        #check that path to hashtable exists
        #if 'DSAPOLADMIN' 
        if self.key not in os.environ.keys():
            print(self.unauthorized)
            return False
        
        #check that hashtable is on disk
        bcryptfile = glob.glob(os.environ[self.key])#'DSAPOLADMIN'])
        if len(bcryptfile) == 0:
            print(self.unauthorized)
            return False

        #read hash table
        f = open(bcryptfile[0],"rb")
        pwhash = f.read()
        f.close()

        #check pw
        pwval = bcrypt.checkpw(self.value.encode(),pwhash)

        #if it matches, update the hashtable
        if pwval:
            f = open(bcryptfile[0],"wb")
            f.write(bcrypt.hashpw(self.value.encode(),bcrypt.gensalt()))
            f.close()
        else:
            print(self.incorrect)#"Incorrect password, try again")
        return pwval
