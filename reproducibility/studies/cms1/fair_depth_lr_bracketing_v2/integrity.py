#!/usr/bin/env python3
import protocol as p
if __name__=='__main__':
    p.verify_sources();print('SOURCE INTEGRITY: PASS; protocol SHA256='+p.PROTOCOL_SHA)
