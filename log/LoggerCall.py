
#import log.MyTextLog  as  myLogger
import MyTextLog  as  myLogger
log = myLogger.serverLogger



def callTest():
    log.logger.info("this is call Test ！！！")
    log.logger.info("this is call Test ！！！")
    log.logger.info("this is call Test ！！！")
    log.logger.info("this is call Test ！！！")
    log.logger.info("this is call Test ！！！")
    log.logger.info("this is call Test ！！！")
    log.logger.info("this is call Test ！！！")
    log.logger.info("this is call Test ！！！")
    log.logger.debug("this is call Test ！！！")

if __name__ == "__main__":
    callTest()