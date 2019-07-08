#!/usr/bin/env python
#
#   Copyright 2019 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""
Tool for updating version files
"""
import argparse
import datetime
import os
import re
import subprocess

SEARCH_PATTERN = r'''^__version__\s+=\s+'([0-9\.]+\w*?)(\+.*)?'$'''
CHART_APP_VERSION = re.compile(r'''appVersion:(.*)''')


def get_git_revision(file, use_short_hash=True):
    """
    Get current GIT revision of file

    :param file: path to file for check
    :type file: str
    :param use_short_hash: return shorten revision id
    :type use_short_hash: bool
    :return: str or None -- revision id
    """
    try:
        directory = file
        if not os.path.isdir(directory):
            directory = os.path.dirname(file)

        revision = subprocess.check_output(['git', 'rev-parse',
                                            '--short' if use_short_hash else '',
                                            'HEAD'],
                                           cwd=directory)
    except subprocess.CalledProcessError:
        return None

    if isinstance(revision, bytes):
        revision = revision.decode('utf-8')

    return revision.strip()


def get_base_version(file):
    """
    Update local version for file

    :param file: path to version file
    :type file: str
    :return: str - base version
    """
    with open(file, 'r') as stream:
        try:
            content = stream.read()
            base_version = re.search(SEARCH_PATTERN, content, flags=re.MULTILINE)
        except Exception as err:
            raise Exception('Can\'t get version from version string')

    if not base_version:
        raise Exception('Empty search results for pattern {!r} in {!r}'.format(SEARCH_PATTERN, content))

    return base_version.group(1)


def patch_helm_charts(charts_directory, build_version):
    """
    """

    for folder in os.listdir(charts_directory):
        path = os.path.join(charts_directory, folder)
        chart_file = os.path.join(path, 'Chart.yaml')
        if os.path.isdir(path) and os.path.isfile(chart_file):
            chart_name = os.path.basename(path)

            with open(chart_file, 'r') as chart_read_stream:
                chart_data = chart_read_stream.read()

            chart_new_data = CHART_APP_VERSION.sub('appVersion: {}'.format(build_version), chart_data)

            with open(chart_file, 'w') as chart_write_stream:
                chart_write_stream.write(chart_new_data)


def work(args, version_file):
    """
    Set version and build metadata to version file

    :param version_file: path to a version file
    :param args: arguments
    :return: None
    """

    if not os.path.exists(version_file) or not os.path.isfile(version_file):
        raise Exception('Cannot find version file: %s' % version_file)

    if not args.git_revision:
        git_revision = get_git_revision(version_file, not args.use_full_commit_id)
        if not git_revision:
            git_revision = '0000'
    else:
        git_revision = args.git_revision

    build_id = args.build_id
    build_user = args.build_user
    date_string = args.date_string

    if not args.build_version:
        local_version_string = '%s.%s.%s' % (date_string, str(build_id), git_revision)
        build_version = '{}-{}'.format(get_base_version(version_file), local_version_string)
    else:
        build_version = args.build_version

    return build_version


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Version file updater (adds time, build id, commit id to version)')
    parser.add_argument('build_id', type=int, help='Set build id')
    parser.add_argument('build_user', type=str, help='Set build user')
    parser.add_argument('date_string', type=str, help='Set date string')
    parser.add_argument('--build-version', type=str, help='Explicitly specify new Legion build version')
    parser.add_argument('--use-full-commit-id', action='store_true', help='Use full git sha commits')
    parser.add_argument('--git-revision', type=str, help='Set git revision')

    args = parser.parse_args()

    try:
        build_version = work(args, os.path.abspath('__version__.py'))
        patch_helm_charts('helms', build_version)
        print(build_version)
    except KeyboardInterrupt:
        print('Interrupt')
        exit(2)
    except Exception as exception:
        print('Exception')
        print(exception)
        exit(3)