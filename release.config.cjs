/**
 * @type {import('semantic-release').GlobalConfig}
 */
module.exports = {
    "branches": [
        "master"
        ],
    "plugins": [
        "@semantic-release/commit-analyzer",
        "@semantic-release/release-notes-generator",
        [
            "@semantic-release/exec",
            {
                "prepareCmd": "echo 0"
            }
        ],
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