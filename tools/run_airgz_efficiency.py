#!/usr/bin/env python3
"""Current manuscript efficiency entry; requires a current-protocol checkpoint."""
import sys
from run import main

if __name__ == '__main__':
    sys.argv[1:1] = ['--task', 'efficiency', '--data', 'AIR_GZ']
    main()
