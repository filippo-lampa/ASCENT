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
        const Forwarder = await ethers.getContractFactory('Forwarder'); 
        forwarder = await upgrades.deployProxy(Forwarder, [eurftoken.target], { initializer: 'initialize' }); 
    }); 

    describe('registerRequestType', function () { 
        it('should not revert when typeName contains valid character "A"', async function () { 
            const validTypeName = 'ValidTypeA'; 
            const validTypeSuffix = 'suffix'; 
            await expect(forwarder.connect(owner).registerRequestType(validTypeName, validTypeSuffix)) 
                .to.not.be.reverted; 
        }); 
    }); 
});