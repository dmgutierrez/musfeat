import coloredlogs, logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)



