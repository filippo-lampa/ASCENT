module.exports = {
  project: {
    buildDir: "auto",
    contractsDir: "auto",
    testDir: "auto",
    skipContracts: ["interfaces", "mock", "test"],
    skipTests: [],
    testingFramework: "auto",
  },
  mutationTesting: {
    coverage: {
      enabled: false,
      reduceMutants: false,
      maxMutants: 100000
    },
    testingTimeOutInSec: 500,
    resultsHistory: true
  }
}