#!/bin/bash 

HOSTNAME=zksync-fuzzing

rsync -av -e ssh --exclude=target --exclude=.git --exclude=compliance-tests . $HOSTNAME:zksync
