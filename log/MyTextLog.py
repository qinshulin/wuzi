
import logging
from logging import handlers
import socket
import time

hostname = socket.gethostname()

class Logger(object):
    level_relations={
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warnging':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='M',backCount=0,fmt = '%(asctime)s - [%(filename)-25s(%(lineno)-5d)] - %(levelname)s: %(message)s'):
        timestr = time.strftime('%F-%T',time.localtime()).replace(':','-')
        filename = filename +'_'+timestr+'.log'
        self.logger = logging.getLogger(filename)
        format_str  = logging.Formatter(fmt) #设置日志格式
        self.logger.setLevel(self.level_relations.get(level)) #设置日志格式

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)

        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器

        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)

 #加hostname
serverLogger = Logger('./log/ins-vision-frontend-hostname-' + str(hostname), level='debug', when='D')
#serverLogger = Logger('./export/Logs/ins-vision-frontend-hostname-'+str(hostname), level='debug', when='D')
# serverLogger = Logger('D:\Develop\jd_prj\ins-bdg-vision-mq', level='debug', when='D')
