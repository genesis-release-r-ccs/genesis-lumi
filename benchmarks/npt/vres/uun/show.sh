#!/bin/bash -e

grep dynamics p008.node001* | awk '{print $4}'
grep dynamics p008.node002* | awk '{print $4}'
grep dynamics p008.node004* | awk '{print $4}'
grep dynamics p008.node008* | awk '{print $4}'

grep dynamics p016.node002* | awk '{print $4}'
grep dynamics p016.node004* | awk '{print $4}'
grep dynamics p016.node008* | awk '{print $4}'
grep dynamics p016.node016* | awk '{print $4}'

grep dynamics p032.node004* | awk '{print $4}'
grep dynamics p032.node008* | awk '{print $4}'
grep dynamics p032.node016* | awk '{print $4}'
grep dynamics p032.node032* | awk '{print $4}'

grep dynamics p064.node008* | awk '{print $4}'
grep dynamics p064.node016* | awk '{print $4}'
grep dynamics p064.node032* | awk '{print $4}'
grep dynamics p064.node064* | awk '{print $4}'

grep dynamics p128.node016* | awk '{print $4}'
grep dynamics p128.node032* | awk '{print $4}'
grep dynamics p128.node064* | awk '{print $4}'
grep dynamics p128.node128* | awk '{print $4}'

grep dynamics p256.node032* | awk '{print $4}'
grep dynamics p256.node064* | awk '{print $4}'
grep dynamics p256.node128* | awk '{print $4}'
grep dynamics p256.node256* | awk '{print $4}'

grep dynamics p512.node064* | awk '{print $4}'
grep dynamics p512.node128* | awk '{print $4}'
grep dynamics p512.node256* | awk '{print $4}'
grep dynamics p512.node512* | awk '{print $4}'
