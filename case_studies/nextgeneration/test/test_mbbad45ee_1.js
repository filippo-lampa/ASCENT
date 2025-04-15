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
        it('should revert when typeName starts with "("', async function () {
            await expect(forwarder.connect(owner).registerRequestType('(', 'suffix')).to.be.revertedWith("NGEUR Forwarder: invalid typename");
        });

        it('should revert when typeName starts with ")"', async function () {
            await expect(forwarder.connect(owner).registerRequestType(')', 'suffix')).to.be.revertedWith("NGEUR Forwarder: invalid typename");
        });
    });
});