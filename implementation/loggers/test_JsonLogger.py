from test.LoggedTestCase import LoggedTestCase
from aienvs.loggers.JsonLogger import JsonLogger
from unittest.mock import Mock
import io


class test_JsonLogger(LoggedTestCase):
    
    def    test_log(self):
        logoutput = io.StringIO("episode output log")
        logger = JsonLogger(logoutput)
        
        data = {'done':True, 'actions':None}
        datajson = '{"done": true, "actions": null}\n'
        
        logger.notifyChange(data)

        self.assertEqual(datajson, logoutput.getvalue())

    def    test_log_multiline(self):
        logoutput = io.StringIO("episode output log")
        logger = JsonLogger(logoutput)
        
        data1 = {'done':False, 'actions':'action1'}
        data2 = {'done':True, 'actions':'action2'}
        datajson = '{"done": false, "actions": "action1"}\n{"done": true, "actions": "action2"}\n'
        
        logger.notifyChange(data1)
        logger.notifyChange(data2)

        self.assertEqual(datajson, logoutput.getvalue())

