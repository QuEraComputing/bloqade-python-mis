# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:50:28 2023

@author: jwurtz
"""
from abc import abstractmethod
import json


class record(dict):
    """
    Record and load data to a simple data structure.
    Data is a dictionary keyed by the string representation of each class, with
    the values being the string method of that class.
    """

    @abstractmethod
    def put(self, inclass):
        """
        Add data to the record
        """

    @abstractmethod
    def get(self, inclass):
        """
        Fetch data from the record
        """

    @abstractmethod
    def save(self):
        """
        Serialize record to file
        """

    @staticmethod
    @abstractmethod
    def load():
        """
        Load data from file
        """

    def __str__(self):
        return json.dumps(self)


class null_record(record):
    """
    A record object that does nothing.
    """

    def put(self, inclass):
        pass

    def get(self, inclass):
        return ""

    def save(self):
        pass

    @staticmethod
    def load(self):
        return null_record()


class local_record(record):
    """
    A record that is saved to a local file system
    """

    def __init__(self, filename: str, logfrequency: int = 1e99):
        self.filename = filename
        self.logfrequency = 1e99
        self.ctr = 0

    def put(self, inclass):
        if isinstance(inclass, type):
            signature = inclass
        else:
            signature = inclass.__class__

        self[str(signature)] = str(inclass)

        self.ctr += 1
        if self.ctr % self.logfrequency == self.logfrequency - 1:
            self.save()

    def get(self, inclass):
        if isinstance(inclass, type):
            signature = inclass
        else:
            signature = inclass.__class__

        return self[str(signature)]

    def save(self):
        with open(self.filename, "w") as io:
            json.dump(self, io)

    @staticmethod
    def load(filename: str):
        dout = local_record(filename)
        with open(filename, "r") as io:
            payload = json.load(io)
            dout.update(payload.items())
        return dout


class AWS_record(record):
    """
    A record that is saved to an AWS S3
    """

    def __init__(self, s3_instance):
        raise NotImplementedError("Not implemented!")

    def pull(self) -> local_record:
        pass
