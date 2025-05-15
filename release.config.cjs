/**
 * @type {import('semantic-release').GlobalConfig}
 */
module.exports = {
    "branches": [
        "master"
        ],
    "prepare": [
        {
            "cmd": "sed -i \"s/^version\\s*=\\s*'.*'/version = '${nextRelease.version}'/\" pyproject.toml"
        }
        ],
    "plugins": [
        "@semantic-release/commit-analyzer",
        "@semantic-release/release-notes-generator",
        [
            "@semantic-release/changelog",
            {
            "changelogFile": "CHANGELOG.md",
            "changelogTitle": "# Changelog\n\nAll notable changes to this project will be documented in this file."
            }
        ],
        "@semantic-release/github",
        [
            "@semantic-release/git",
            {
            "assets": [
                "pyproject.toml",
                "CHANGELOG.md"
            ],
            "message": "chore(release): version ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}"
            }
        ]
    ]
}