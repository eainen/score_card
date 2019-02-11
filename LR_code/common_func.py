#!/usr/bin/env python
# -*- coding: utf-8 -*-
# common function



import calendar
import os
from time import ctime

import requests


def execute_hql(hql):
    # linux连接hive
    status = os.system(
        "hive -e \"SET mapreduce.job.queuename=root.anti_fraud; \
        SET hive.exec.max.dynamic.partitions = 3000;\
        SET hive.exec.max.dynamic.partitions.pernode=3000;%s\";" % hql)
    return status


def executeHQL_f(sql, file_pwd):
    status = os.system(
        "hive -e \"SET mapreduce.job.queuename=root.anti_fraud; %s \" > %s "
        % (sql, file_pwd))
    return status


def send_alert(tag, alert):
    """ wrap alert message with time and header, then send it to iPortal
        @alert :: string
    """
    return requests.post('http://nfjd-alert.jpushoa.com/v1/alert/',
                         json={
                             "code": 185,
                             "desc": u" ".join(
                                 [tag, u"\napp-stats:", ctime(), u"\n", alert]
                             )
                         })


def add_months(date, months):
    month = date.month - 1 + months
    year = date.year + month / 12
    month = month % 12 + 1
    day = min(date.day, calendar.monthrange(year, month)[1])
    return date.replace(year=year, month=month, day=day)

