const { expect } = require('chai');
const { ethers, upgrades } = require('hardhat');

describe('Forwarder', function () {
    let forwarder;
    let owner;
    let eurftoken;

    beforeEach(async function () {
        [owner] = await ethers.getSigners();
        const EURFToken = await ethers.getContractFactory('EURFToken');
        eurftoken = await upgrades.deployProxy(EURFToken, [], { kind: 'uups', initializer: 'initialize' });
        
        const FORWARDER = await ethers.getContractFactory('Forwarder');
        forwarder = await upgrades.deployProxy(FORWARDER, [eurftoken.target], { initializer: 'initialize' });
    });

    describe('registerRequestType', function () {
        it('should revert when typeName contains invalid characters "(", ")"', async function () {
            const invalidTypeName = "invalid(type)";
            const invalidTypeSuffix = "suffix";

            await expect(forwarder.connect(owner).registerRequestType(invalidTypeName, invalidTypeSuffix))
                .to.be.revertedWith("NGEUR Forwarder: invalid typename");
        });
    });
});