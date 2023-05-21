import sys
import time

from classifier.test_running_code import run_test
from classifier.convert_morlets import convert_morlets

from classifier.config import RUN_TEST, CONVERT_MORLETS, dotdict, test_config


def main():
    if CONVERT_MORLETS:
        convert_morlets()

    if RUN_TEST:
        test_configs = [dotdict()]
        for key, values in test_config.items():
            if isinstance(values, list) and len(values) > 0:
                new_test_configs = []
                for every_test_config in test_configs:
                    for i in range(len(values)):
                        new_config = every_test_config.copy()
                        new_config[key] = values[i]
                        new_test_configs.append(new_config)
                test_configs = new_test_configs.copy()
            else:
                for every_test_config in test_configs:
                    every_test_config[key] = values

        for every_test_config in test_configs:

            # Output properties
            timestamp                           = round(time.time() * 1000)
            every_test_config.checkpoint_prefix = f'autorun-{timestamp}'
            every_test_config.test_json         = f'checkpoints/{every_test_config.checkpoint_prefix}.json'
            every_test_config.test_config_json  = f'checkpoints/{every_test_config.checkpoint_prefix}-config.json'

            run_test(every_test_config)


if __name__ == '__main__':
    sys.exit(main())
