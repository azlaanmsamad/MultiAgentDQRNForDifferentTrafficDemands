from test.LoggedTestCase import LoggedTestCase
from aienvs.loggers.PickleLogger import PickleLogger
from unittest.mock import Mock
import io
import pickle


class test_PickleLogger(LoggedTestCase):
    
    def test_pickle(self):
        """
        Test that shows how to read multiple datas from the dump
        """
        data1 = {1, 2, 3}
        data2 = {4, 5}
        logoutputpickle = io.BytesIO()
        pickle.dump(data1, logoutputpickle)
        pickle.dump(data2, logoutputpickle)
        instream = io.BytesIO(logoutputpickle.getvalue())
        self.assertEquals (data1, pickle.load(instream))
        self.assertEquals (data2, pickle.load(instream))

    def    test_log(self):
        logoutput = io.BytesIO()
        logger = PickleLogger(logoutput)

        # log 1 event        
        data = {'done':True, 'actions':None}
        logger.notifyChange(data)

        # now check what has been logged
        instream = io.BytesIO(logoutput.getvalue())
        self.assertEqual(data, pickle.load(instream))

    def    test_log_multiline(self):
        logoutput = io.BytesIO()
        logger = PickleLogger(logoutput)
        
        # log 2 events
        data1 = {'done':True, 'actions':None}
        data2 = {'done':True, 'actions':'action2'}
        logger.notifyChange(data1)
        logger.notifyChange(data2)

        # now check what has been logged
        instream = io.BytesIO(logoutput.getvalue())
        self.assertEqual(data1, pickle.load(instream))
        self.assertEqual(data2, pickle.load(instream))

